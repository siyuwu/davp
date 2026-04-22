"""
Simulation Script for DAVP Benchmark Preparation

This script demonstrates the pipeline for preparing simulated patient samples
from rare disease benchmarks (e.g., UDN) by:
1. Loading benchmark data with known pathogenic variants
2. Converting gene-level annotations to specific variants using ClinVar
3. Inserting pathogenic variants into background genomes (1000 Genomes VCFs)
4. Running variant filtering (e.g., Exomiser) to produce analysis-ready files

NOTE: This is a demonstration script for the paper release. Full implementation
would require additional dependencies and data files not included in this repo.

Directory Structure:
    davp/
    ├── data/
    │   ├── clinvar/           # ClinVar VCF files (user must provide)
    │   │   └── clinvar_*.vcf.gz
    │   └── input/             # Generated sample JSONLs (created by this script)
    │       └── <sample_name>.jsonl
    ├── benchmarks/
    │   └── udn.jsonl          # Benchmark file with known variants
    └── simulation.py          # This script
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
from pathlib import Path
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
import unicodedata
import argparse
import logging
import random
import json
import glob
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAVP_ROOT = Path(__file__).parent
DAVP_DATA_DIR = DAVP_ROOT / "data"
CLINVAR_DIR = DAVP_DATA_DIR / "clinvar"
BENCHMARKS_DIR = DAVP_ROOT / "benchmarks"
INPUT_DIR = DAVP_DATA_DIR / "input"


# =============================================================================
# ClinVar Constants and Mappings
# =============================================================================

CLINVAR_CLNSIG_MAP = {
    'Uncertain_significance': ['uncertain'],
    'Conflicting_classifications_of_pathogenicity': ['conflicting'],
    'Benign': ['benign'],
    'Likely_benign': ['likely_benign'],
    'Benign/Likely_benign': ['benign', 'likely_benign'],
    'Pathogenic': ['pathogenic'],
    'Pathogenic/Likely_risk_allele': ['pathogenic'],
    'Pathogenic/Likely_pathogenic': ['pathogenic', 'likely_pathogenic'],
    'Pathogenic/Likely_pathogenic/Pathogenic_low_penetrance': ['pathogenic', 'likely_pathogenic', 'low_penetrance'],
    'Likely_pathogenic': ['likely_pathogenic'],
    'Likely_pathogenic_low_penetrance': ['likely_pathogenic', 'low_penetrance'],
    'Likely_pathogenic/Likely_risk_allele': ['likely_pathogenic'],
}

CLINVAR_REVSTAT_POINTS = {
    'criteria_provided_conflicting_classifications': 0,
    'criteria_provided_multiple_submitters_no_conflicts': 3,
    'criteria_provided_single_submitter': 2,
    'no_assertion_criteria_provided': 1,
    'no_classification_for_the_single_variant': 0,
    'no_classification_provided': 0,
    'no_classifications_from_unflagged_records': 0,
    'practice_guideline': 5,
    'reviewed_by_expert_panel': 4,
}

VALID_CONTIGS = [str(i) for i in range(1, 23)] + ['X', 'Y']


# =============================================================================
# Data Models
# =============================================================================

class BenchmarkEntry(BaseModel):
    """A single entry from the benchmark file representing a known pathogenic variant."""
    contig: str
    pos: int
    ref: str
    alt: str
    clnsig: list[str]
    clnrevstat: int
    genes: list[str]
    diseases: list[str]
    origins: list[str]
    hpos: list[str]
    candidate_allele_ids: list[str]
    epicrisis: str = ""
    sample_name: str = ""
    type: str = "pathogenic_noisy"


class ClinVarVariant(BaseModel):
    """A variant from ClinVar database."""
    contig: str
    pos: int
    ref: str
    alt: str
    clnsig: list[str]
    clnrevstat: int
    genes: list[str]
    diseases: list[str]
    fuzzy_score: Optional[float] = None


class SampleVariant(BaseModel):
    """A variant in the final sample JSONL format (Exomiser-filtered)."""
    model_config = ConfigDict(populate_by_name=True)
    
    CHROM: int | str
    POS: int
    REF: str
    ALT: str
    QUAL: Optional[float] = None
    GT: str = "0/1"
    Databases: list[str] = Field(default_factory=list)
    Gene_Name: list[str] = Field(default_factory=list, alias="Gene Name")
    ClinVar_SIG: list[str] = Field(default_factory=list, alias="ClinVar SIG")
    gnomAD_AF: Optional[float] = Field(default=None, alias="gnomAD AF")
    ACMG: list = Field(default_factory=list)
    OMIM: bool = False


# =============================================================================
# ClinVar Processing Utilities
# =============================================================================

def _canonicalize_disease_name(disease: str) -> str:
    """
    Normalize disease name for fuzzy matching.
    Removes common suffixes, converts roman numerals, normalizes whitespace.
    """
    if pd.isna(disease):
        return ""
    s = unicodedata.normalize("NFKD", str(disease)).lower()
    s = re.sub(r"[\[\]_/,()-]+", " ", s)
    s = re.sub(r"\b(syndrome|disorder|disease|type)\b", "", s)
    s = re.sub(
        r"\b(x|ix|viii|vii|vi|iv|iii|ii|i)\b",
        lambda m: {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
                   "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}[m.group()],
        s
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_clinvar_dataset(clinvar_path: Path) -> pd.DataFrame:
    """
    Load a processed ClinVar TSV file.
    
    Expected format (tab-separated):
        contig, pos, ref, alt, clnsig, clnrevstat, genes, diseases, origins, hpos, mondo, omim, orpha
    
    Args:
        clinvar_path: Path to the processed ClinVar TSV (gzipped)
        
    Returns:
        DataFrame with parsed ClinVar variants
    """
    if not clinvar_path.exists():
        logger.warning(f"ClinVar file not found: {clinvar_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(
        clinvar_path,
        compression="gzip" if str(clinvar_path).endswith(".gz") else None,
        sep="\t",
        dtype={
            "contig": str,
            "pos": int,
            "ref": str,
            "alt": str,
            "clnsig": str,
            "clnrevstat": int,
            "genes": str,
            "diseases": str,
            "origins": str,
        }
    )
    
    # Parse comma-separated fields into lists
    df["clnsig"] = df["clnsig"].str.split(",")
    df["genes"] = df["genes"].str.split(",")
    df["diseases"] = df["diseases"].str.split(",")
    
    return df


def filter_clinvar_by_gene(clinvar_df: pd.DataFrame, gene_symbol: str) -> pd.DataFrame:
    """Filter ClinVar variants to those affecting a specific gene."""
    if clinvar_df.empty:
        return clinvar_df
    return clinvar_df[clinvar_df["genes"].apply(lambda x: gene_symbol in x if isinstance(x, list) else False)]


def filter_clinvar_by_pathogenicity(clinvar_df: pd.DataFrame, 
                                     significance: Literal["pathogenic", "uncertain", "benign"]) -> pd.DataFrame:
    """
    Filter ClinVar variants by clinical significance.
    
    Args:
        clinvar_df: ClinVar DataFrame
        significance: One of 'pathogenic', 'uncertain', or 'benign'
    """
    if clinvar_df.empty:
        return clinvar_df
    
    if significance == "pathogenic":
        return clinvar_df[clinvar_df["clnsig"].apply(
            lambda x: "pathogenic" in x or "likely_pathogenic" in x if isinstance(x, list) else False
        )]
    elif significance == "uncertain":
        return clinvar_df[clinvar_df["clnsig"].apply(
            lambda x: "uncertain" in x or "conflicting" in x if isinstance(x, list) else False
        )]
    elif significance == "benign":
        return clinvar_df[clinvar_df["clnsig"].apply(
            lambda x: "benign" in x or "likely_benign" in x if isinstance(x, list) else False
        )]
    return clinvar_df


def filter_clinvar_by_disease(clinvar_df: pd.DataFrame, 
                               disease_name: str, 
                               fuzzy_threshold: float = 0.85) -> pd.DataFrame:
    """
    Filter ClinVar variants by disease name using fuzzy matching.
    
    Args:
        clinvar_df: ClinVar DataFrame
        disease_name: Target disease name to match
        fuzzy_threshold: Minimum similarity score (0-1)
    """
    if clinvar_df.empty:
        return clinvar_df
    
    query = _canonicalize_disease_name(disease_name)
    
    normalized_diseases = []
    row_indices = []
    
    for idx, diseases_list in clinvar_df["diseases"].items():
        if isinstance(diseases_list, list):
            for disease in diseases_list:
                normalized_diseases.append(_canonicalize_disease_name(disease))
                row_indices.append(idx)
    
    if not normalized_diseases:
        return clinvar_df.iloc[:0]
    
    matches = process.extract(
        query, 
        normalized_diseases, 
        scorer=fuzz.token_set_ratio, 
        score_cutoff=fuzzy_threshold * 100
    )
    
    if not matches:
        return clinvar_df.iloc[:0]
    
    keep_rows = set(row_indices[m[2]] for m in matches)
    result = clinvar_df.loc[sorted(keep_rows)].copy()
    
    # Add fuzzy match score for ranking
    result["fuzzy_score"] = result.index.map(
        lambda idx: max(
            fuzz.token_set_ratio(query, _canonicalize_disease_name(d))
            for d in (result.loc[idx, "diseases"] if isinstance(result.loc[idx, "diseases"], list) else [])
        ) / 100.0 if idx in result.index else 0
    )
    
    return result.sort_values(by="fuzzy_score", ascending=False)


def gene_to_variant(gene_symbol: str, 
                    disease_name: str,
                    clinvar_df: pd.DataFrame) -> Optional[ClinVarVariant]:
    """
    Convert a gene symbol to a specific ClinVar variant.
    
    This implements a cascading strategy to find the best matching variant:
    1. Pathogenic variants matching both gene AND disease
    2. Uncertain significance variants matching both gene AND disease
    3. Pathogenic variants matching gene only
    4. Uncertain significance variants matching gene only
    5. Benign variants as fallback
    
    Args:
        gene_symbol: HGNC gene symbol (e.g., 'BRCA1')
        disease_name: Disease name for matching
        clinvar_df: Full ClinVar DataFrame
        
    Returns:
        ClinVarVariant if found, None otherwise
    """
    gene_filtered = filter_clinvar_by_gene(clinvar_df, gene_symbol)
    
    if gene_filtered.empty:
        logger.debug(f"No ClinVar variants found for gene: {gene_symbol}")
        return None
    
    # Strategy cascade
    strategies = [
        ("pathogenic", True),   # Pathogenic + disease match
        ("uncertain", True),    # VUS + disease match
        ("pathogenic", False),  # Pathogenic only
        ("uncertain", False),   # VUS only
        ("benign", True),       # Benign + disease match (fallback)
        ("benign", False),      # Benign only (last resort)
    ]
    
    for significance, use_disease in strategies:
        candidates = filter_clinvar_by_pathogenicity(gene_filtered, significance)
        
        if use_disease and not candidates.empty:
            candidates = filter_clinvar_by_disease(candidates, disease_name)
        
        if not candidates.empty:
            selected = candidates.sample(1).iloc[0]
            return ClinVarVariant(
                contig=selected["contig"],
                pos=selected["pos"],
                ref=selected["ref"],
                alt=selected["alt"],
                clnsig=selected["clnsig"] if isinstance(selected["clnsig"], list) else [],
                clnrevstat=selected["clnrevstat"],
                genes=selected["genes"] if isinstance(selected["genes"], list) else [],
                diseases=selected["diseases"] if isinstance(selected["diseases"], list) else [],
                fuzzy_score=selected.get("fuzzy_score")
            )
    
    return None


# =============================================================================
# VCF Manipulation (Placeholder Implementation)
# =============================================================================

def insert_variant_into_vcf(vcf_path: Path, 
                            variant: ClinVarVariant, 
                            output_path: Path) -> bool:
    """
    Insert a variant into a VCF file.
    
    NOTE: This is a placeholder implementation for demonstration purposes.
    A full implementation would use pysam or cyvcf2 to properly manipulate VCF files.
    
    Args:
        vcf_path: Path to input VCF (background genome from 1000 Genomes)
        variant: Variant to insert
        output_path: Path for output VCF
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[PLACEHOLDER] Would insert variant chr{variant.contig}:{variant.pos}{variant.ref}>{variant.alt}")
    logger.info(f"  Input VCF: {vcf_path}")
    logger.info(f"  Output VCF: {output_path}")
    
    # In a full implementation:
    # 1. Read VCF header and records using pysam.VariantFile
    # 2. Create new record with variant information
    # 3. Insert at correct position (sorted by CHROM, POS)
    # 4. Write to output file
    
    # For demonstration, we just log the operation
    return True


def download_1000_genomes_vcf(sample_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download a background genome VCF from 1000 Genomes Project.
    
    NOTE: This is a placeholder implementation. A full implementation would:
    1. Query the 1000 Genomes FTP/HTTP server
    2. Download the per-sample VCF or extract from population VCF
    3. Index the VCF using tabix
    
    Args:
        sample_id: 1000 Genomes sample ID (e.g., 'HG00126')
        output_dir: Directory to save the VCF
        
    Returns:
        Path to downloaded VCF, or None if download failed
    """
    logger.info(f"[PLACEHOLDER] Would download VCF for sample: {sample_id}")
    logger.info(f"  1000 Genomes FTP: ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/...")
    
    # Return None to indicate this is a placeholder
    return None


# =============================================================================
# Sample Preparation
# =============================================================================

def vcf_to_sample_jsonl(vcf_path: Path, output_path: Path) -> bool:
    """
    Convert a VCF file to sample JSONL format after Exomiser filtering.
    
    NOTE: This is a placeholder implementation for demonstration purposes.
    A full implementation would:
    1. Run Exomiser variant prioritization on the VCF
    2. Parse Exomiser output
    3. Convert to the JSONL format expected by DAVP
    
    Expected output format (one JSON object per line):
    {
        "CHROM": 15,
        "POS": 42360096,
        "REF": "C",
        "ALT": "A",
        "QUAL": null,
        "GT": "1/1",
        "Databases": ["rs758058910", "COSV58824256"],
        "Gene Name": ["CAPN3"],
        "1KG Frequency": null,
        "ClinVar SIG": ["pathogenic"],
        "DP": null,
        "AD": [-1, -1],
        "gnomAD AF": 0.00000205,
        "gnomADg AF": null,
        "ACMG": ["Uncertain significance", ["BP4", "PM2", "PP5"]],
        "OMIM": true,
        "Libra Pathogenicity": 0.18065121,
        "Sorting Score": 7335,
        "ACMG Severity": 3,
        "Pathogenicity ClinVar Severity": 10,
        "Alpha Missense Score": 0.9259,
        "Alpha Missense Prediction": "P",
        "SIFT": 0.11,
        "Polyphen": 0.021,
        "CADD": 15.47,
        "REVEL": 0.325,
        "SpliceAI": ["CAPN3", "0.00", "0.00", "0.00", "0.00", "18", "-21", "-4", "18"],
        "DANN": 0.97951211,
        "MetalR": 0.371,
        "Gene Phenotype": 1
    }
    
    Args:
        vcf_path: Path to input VCF (with inserted pathogenic variant)
        output_path: Path for output JSONL
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[PLACEHOLDER] Would convert VCF to JSONL:")
    logger.info(f"  Input VCF: {vcf_path}")
    logger.info(f"  Output JSONL: {output_path}")
    logger.info("  Steps:")
    logger.info("    1. Run Exomiser on VCF")
    logger.info("    2. Parse variant annotations")
    logger.info("    3. Write JSONL format")
    
    return True


def load_benchmark(benchmark_path: Path) -> list[BenchmarkEntry]:
    """
    Load benchmark file containing known pathogenic variants.
    
    Args:
        benchmark_path: Path to benchmark JSONL file
        
    Returns:
        List of BenchmarkEntry objects
    """
    entries = []
    with open(benchmark_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(BenchmarkEntry(**data))
    logger.info(f"Loaded {len(entries)} entries from {benchmark_path}")
    return entries


def prepare_sample(benchmark_entry: BenchmarkEntry,
                   clinvar_df: pd.DataFrame,
                   output_dir: Path) -> dict:
    """
    Prepare a single simulated sample from a benchmark entry.
    
    This function orchestrates the full simulation pipeline:
    1. Download background genome from 1000 Genomes
    2. Convert gene annotations to specific variants using ClinVar
    3. Insert pathogenic variant into background VCF
    4. Run variant filtering to produce JSONL output
    
    Args:
        benchmark_entry: Entry from benchmark file
        clinvar_df: ClinVar DataFrame for gene-to-variant conversion
        output_dir: Directory for output files
        
    Returns:
        Dictionary with sample preparation results
    """
    sample_name = benchmark_entry.sample_name
    logger.info(f"Preparing sample: {sample_name}")
    
    result = {
        "sample_name": sample_name,
        "status": "pending",
        "pathogenic_variant": None,
        "candidate_variants": [],
        "output_path": None
    }
    
    # Step 1: Get pathogenic variant (from benchmark or ClinVar lookup)
    pathogenic_variant = ClinVarVariant(
        contig=benchmark_entry.contig,
        pos=benchmark_entry.pos,
        ref=benchmark_entry.ref,
        alt=benchmark_entry.alt,
        clnsig=benchmark_entry.clnsig,
        clnrevstat=benchmark_entry.clnrevstat,
        genes=benchmark_entry.genes,
        diseases=benchmark_entry.diseases
    )
    result["pathogenic_variant"] = pathogenic_variant.model_dump()
    
    # Step 2: Convert candidate genes to variants
    for allele_id in benchmark_entry.candidate_allele_ids[:5]:  # Limit for demo
        # Parse allele ID format: "chr16:90040510C>T"
        match = re.match(r"chr(\w+):(\d+)([ACGT]+)>([ACGT]+)", allele_id)
        if match:
            result["candidate_variants"].append({
                "allele_id": allele_id,
                "contig": match.group(1),
                "pos": int(match.group(2)),
                "ref": match.group(3),
                "alt": match.group(4)
            })
    
    # Step 3: Create output directory and prepare sample
    sample_output = output_dir / f"{sample_name}.jsonl"
    result["output_path"] = str(sample_output)
    
    # In a full implementation, we would:
    # - Download background VCF
    # - Insert variants
    # - Run filtering
    # - Generate JSONL
    
    logger.info(f"  Pathogenic variant: chr{pathogenic_variant.contig}:{pathogenic_variant.pos}")
    logger.info(f"  Candidate variants: {len(result['candidate_variants'])}")
    logger.info(f"  Output path: {sample_output}")
    
    result["status"] = "placeholder_complete"
    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare simulated samples from rare disease benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare a single sample
    python simulation.py --benchmark benchmarks/udn.jsonl --sample HG01948
    
    # Prepare all samples from benchmark
    python simulation.py --benchmark benchmarks/udn.jsonl --all
    
    # Use specific ClinVar version
    python simulation.py --benchmark benchmarks/udn.jsonl --clinvar data/clinvar/clinvar_2024.tsv.gz
"""
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        type=Path,
        default=BENCHMARKS_DIR / "udn.jsonl",
        help="Path to benchmark JSONL file"
    )
    parser.add_argument(
        "--sample", "-s",
        type=str,
        help="Specific sample name to prepare"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Prepare all samples from benchmark"
    )
    parser.add_argument(
        "--clinvar", "-c",
        type=Path,
        help="Path to processed ClinVar TSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=INPUT_DIR,
        help="Output directory for sample JSONLs"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Maximum number of samples to prepare (with --all)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output}")
    
    # Load benchmark
    if not args.benchmark.exists():
        logger.error(f"Benchmark file not found: {args.benchmark}")
        return 1
    
    entries = load_benchmark(args.benchmark)
    
    # Load ClinVar if provided
    clinvar_df = pd.DataFrame()
    if args.clinvar and args.clinvar.exists():
        clinvar_df = load_clinvar_dataset(args.clinvar)
        logger.info(f"Loaded {len(clinvar_df)} ClinVar variants")
    else:
        logger.warning("ClinVar not provided or not found - gene-to-variant conversion will be limited")
    
    # Filter entries
    if args.sample:
        entries = [e for e in entries if e.sample_name == args.sample]
        if not entries:
            logger.error(f"Sample not found in benchmark: {args.sample}")
            return 1
    elif args.all:
        entries = entries[:args.limit]
    else:
        # Default: just show info
        logger.info(f"\nBenchmark contains {len(entries)} samples")
        logger.info("Use --sample NAME to prepare a specific sample")
        logger.info("Use --all to prepare all samples (limited by --limit)")
        return 0
    
    # Prepare samples
    results = []
    for entry in entries:
        result = prepare_sample(entry, clinvar_df, args.output)
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Prepared {len(results)} samples")
    for r in results:
        logger.info(f"  {r['sample_name']}: {r['status']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
