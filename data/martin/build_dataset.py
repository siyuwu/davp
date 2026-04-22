"""
Extract ground truth records from phenopacket JSON files and write a
dataset.jsonl compatible with davp.py's load_answers() / get_answer_for_sample().

Usage:
    python data/martin/build_dataset.py

Output:
    data/input/martin_dataset.jsonl
"""

import json
import glob
from pathlib import Path

PHENOPACKETS_DIR = Path(__file__).parent / "phenopackets"
OUTPUT_PATH = Path(__file__).parent.parent / "input" / "martin_dataset.jsonl"


def extract_record(phenopacket: dict) -> dict:
    """Convert one phenopacket dict into a dataset.jsonl record."""

    sample_name = phenopacket["id"]

    # --- HPO terms (active only, not excluded) ---
    hpos = []
    hpo_inputs = []
    for feature in phenopacket.get("phenotypicFeatures", []):
        if feature.get("excluded"):
            continue
        term = feature["type"]
        hpos.append(term["id"])
        hpo_inputs.append({
            "id": term["id"],
            "name": term["label"],
            "definition": "",   # not present in phenopackets
            "synonym": [],      # not present in phenopackets
        })

    # --- Disease labels ---
    diseases = [
        d["term"]["label"]
        for d in phenopacket.get("diseases", [])
        if "term" in d
    ]

    # --- Causative variant (first CAUSATIVE genomic interpretation) ---
    contig = pos = ref = alt = None
    genes = []
    clnsig = []
    for interpretation in phenopacket.get("interpretations", []):
        for gi in interpretation.get("diagnosis", {}).get("genomicInterpretations", []):
            if gi.get("interpretationStatus") != "CAUSATIVE":
                continue
            vi = gi.get("variantInterpretation", {})
            vd = vi.get("variationDescriptor", {})

            # Gene
            gene_symbol = vd.get("geneContext", {}).get("symbol")
            if gene_symbol:
                genes.append(gene_symbol)

            # ACMG classification → clnsig
            acmg = vi.get("acmgPathogenicityClassification", "")
            if acmg:
                clnsig.append(acmg.lower())

            # Genomic coordinates
            vcf = vd.get("vcfRecord", {})
            raw_chrom = vcf.get("chrom", "")
            contig = raw_chrom.replace("chr", "")
            pos = int(vcf["pos"]) if vcf.get("pos") else None
            ref = vcf.get("ref")
            alt = vcf.get("alt")
            break  # use first causative variant only

    return {
        "sample_name": sample_name,
        "contig": contig,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "genes": genes,
        "diseases": diseases,
        "clnsig": clnsig,
        "clnrevstat": 0,    # not available in phenopackets
        "origins": "",      # not available in phenopackets
        "hpos": hpos,
        "hpo_inputs": hpo_inputs,
        "epicrisis": "",    # not available in phenopackets
    }


def build_dataset(phenopackets_dir: Path, output_path: Path) -> int:
    """Read all phenopacket JSON files and write dataset.jsonl. Returns record count."""
    json_files = sorted(phenopackets_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {phenopackets_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for path in json_files:
        with open(path) as f:
            phenopacket = json.load(f)
        record = extract_record(phenopacket)
        records.append(record)
        print(f"  Extracted: {record['sample_name']} | gene={record['genes']} | "
              f"variant={record['contig']}:{record['pos']}{record['ref']}>{record['alt']}")

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(records)} records to {output_path}")
    return len(records)


if __name__ == "__main__":
    build_dataset(PHENOPACKETS_DIR, OUTPUT_PATH)
