import os
import json
from typing import Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from google import genai
from google.genai import types

T = TypeVar('T', bound=BaseModel)


class Response:
    def __init__(self, content: str, structured_output: Optional[BaseModel] = None):
        self.content = content
        self.structured_output = structured_output
        self.message = type('Message', (), {'structured_output': structured_output})()


class BatchResponse:
    def __init__(self, responses: list[Response]):
        self.responses = responses


def generate(prompt: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.0, max_tokens: int = 8192) -> str:
    """Generate a single response from Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    return response.text


def batch_generate(
    prompts: list[str],
    response_model: Optional[type[BaseModel]] = None,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_workers: int = 10
) -> BatchResponse:
    """Generate responses in parallel using ThreadPoolExecutor."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    def _generate_one(prompt: str, index: int) -> tuple[int, Response]:
        try:
            if response_model:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=response_model,
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            content = response.text
            
            # Parse structured output if response_model was provided
            structured_output = None
            if response_model:
                structured_output = response_model.model_validate_json(content)
            
            return index, Response(content, structured_output)
        except Exception as e:
            raise Exception(f"Generation failed: {e}")
    
    # Collect results in order
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, prompt in enumerate(prompts):
            futures[executor.submit(_generate_one, prompt, i)] = i
    
    ordered_results = [None] * len(prompts)
    for future in as_completed(futures):
        index, result = future.result()
        ordered_results[index] = result
    
    return BatchResponse(ordered_results)


class Session:
    def __init__(self, model: str = "gemini-2.0-flash-exp", temperature: float = 0.0, max_tokens: int = 8192, max_workers: int = 10):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
    
    def batch_generate(
        self,
        prompts: list[str],
        response_model: Optional[type[BaseModel]] = None,
        **kwargs
    ) -> BatchResponse:
        return batch_generate(
            prompts,
            response_model=response_model,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            max_workers=kwargs.get("max_workers", self.max_workers)
        )