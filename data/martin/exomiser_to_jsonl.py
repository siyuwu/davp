"""
Convert Exomiser TSV_VARIANT output to DAVP input JSONL format.

Usage:
    python data/martin/exomiser_to_jsonl.py \
        --tsv data/martin/exomiser_results/PMID_15731757_Family_1_I_2.variants.tsv \
        --genes data/martin/exomiser_results/PMID_15731757_Family_1_I_2.genes.tsv \
        --sample PMID_15731757_Family_1_I_2 \
        --top-genes 256

Output:
    data/input/PMID_15731757_Family_1_I_2.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional

OUTPUT_DIR = Path(__file__).parent.parent / "input"


# ---------------------------------------------------------------------------
# Frequency / pathogenicity string parsers
# ---------------------------------------------------------------------------

def parse_kv_string(s: str) -> dict:
    """Parse 'KEY1=val1,KEY2=val2,...' into a dict."""
    result = {}
    if not s or s.strip() == ".":
        return result
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def get_gnomad_exome_af(freq_dict: dict) -> Optional[float]:
    """Return max gnomAD exome AF across all populations."""
    keys = [k for k in freq_dict if k.startswith("GNOMAD_E_")]
    vals = []
    for k in keys:
        try:
            vals.append(float(freq_dict[k]))
        except (ValueError, TypeError):
            pass
    return max(vals) if vals else None


def get_gnomad_genome_af(freq_dict: dict) -> Optional[float]:
    """Return max gnomAD genome AF across all populations."""
    keys = [k for k in freq_dict if k.startswith("GNOMAD_G_")]
    vals = []
    for k in keys:
        try:
            vals.append(float(freq_dict[k]))
        except (ValueError, TypeError):
            pass
    return max(vals) if vals else None


def get_1kg_af(freq_dict: dict) -> Optional[float]:
    keys = [k for k in freq_dict if "THOUSAND_GENOMES" in k or k.startswith("KG")]
    for k in keys:
        try:
            return float(freq_dict[k])
        except (ValueError, TypeError):
            pass
    return None


def get_path_score(path_dict: dict, key_prefix: str) -> Optional[float]:
    for k, v in path_dict.items():
        if k.upper().startswith(key_prefix.upper()):
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# ACMG severity mapping
# ---------------------------------------------------------------------------

ACMG_SEVERITY = {
    "PATHOGENIC": 5,
    "LIKELY_PATHOGENIC": 4,
    "UNCERTAIN_SIGNIFICANCE": 3,
    "LIKELY_BENIGN": 2,
    "BENIGN": 1,
    "NOT_PROVIDED": 0,
}

CLINVAR_SEVERITY = {
    "PATHOGENIC": 10,
    "PATHOGENIC_OR_LIKELY_PATHOGENIC": 9,
    "LIKELY_PATHOGENIC": 8,
    "CONFLICTING_CLASSIFICATIONS_OF_PATHOGENICITY": 5,
    "UNCERTAIN_SIGNIFICANCE": 4,
    "LIKELY_BENIGN": 2,
    "BENIGN_OR_LIKELY_BENIGN": 1,
    "BENIGN": 1,
}


# ---------------------------------------------------------------------------
# Row converter
# ---------------------------------------------------------------------------

def row_to_record(row: dict, gene_rank: int) -> dict:
    contig = row["CONTIG"].replace("chr", "")
    try:
        pos = int(row["START"])
    except (ValueError, TypeError):
        pos = None

    try:
        qual = float(row["QUAL"]) if row.get("QUAL") not in ("", ".", None) else None
    except (ValueError, TypeError):
        qual = None

    # Databases: VCF_ID and RS_ID
    databases = [x for x in [row.get("VCF_ID", ""), row.get("RS_ID", "")] if x and x != "."]

    # Genotype — normalize "0|1" → "0/1" style
    gt = row.get("GENOTYPE", "0/1").replace("|", "/")

    freq_dict = parse_kv_string(row.get("ALL_FREQ", ""))
    path_dict = parse_kv_string(row.get("ALL_PATH", ""))

    gnomad_af = get_gnomad_exome_af(freq_dict)
    gnomad_genome_af = get_gnomad_genome_af(freq_dict)
    kg_freq = get_1kg_af(freq_dict)

    # ClinVar
    clinvar_raw = row.get("CLINVAR_PRIMARY_INTERPRETATION", "")
    clinvar_sig = [clinvar_raw.lower().replace("_", " ")] if clinvar_raw and clinvar_raw != "." else []

    # ACMG
    acmg_class = row.get("EXOMISER_ACMG_CLASSIFICATION", "")
    acmg_evidence_raw = row.get("EXOMISER_ACMG_EVIDENCE", "")
    if acmg_class and acmg_class not in (".", "NOT_PROVIDED", ""):
        acmg_criteria = [e.strip() for e in acmg_evidence_raw.split(",") if e.strip() and e.strip() != "."]
        acmg_label = acmg_class.replace("_", " ").title()
        acmg = [acmg_label, acmg_criteria]
    else:
        acmg = None

    # OMIM
    disease_id = row.get("EXOMISER_ACMG_DISEASE_ID", "")
    omim = "OMIM:" in disease_id if disease_id else False

    # Pathogenicity scores
    revel = get_path_score(path_dict, "REVEL")
    sift = get_path_score(path_dict, "SIFT")
    polyphen = get_path_score(path_dict, "POLYPHEN")
    cadd = get_path_score(path_dict, "CADD")

    # Severity integers for sorting
    acmg_severity = ACMG_SEVERITY.get(acmg_class.upper() if acmg_class else "", 0)
    clinvar_raw_upper = clinvar_raw.upper().replace(" ", "_") if clinvar_raw else ""
    clinvar_severity = CLINVAR_SEVERITY.get(clinvar_raw_upper, 0)

    # Sorting score: higher Exomiser combined score → higher sorting score
    try:
        combined = float(row.get("EXOMISER_GENE_COMBINED_SCORE", 0))
    except (ValueError, TypeError):
        combined = 0.0
    sorting_score = int(combined * 10000)

    # Gene phenotype: 1 if pheno score > 0
    try:
        pheno_score = float(row.get("EXOMISER_GENE_PHENO_SCORE", 0))
    except (ValueError, TypeError):
        pheno_score = 0.0
    gene_phenotype = 1 if pheno_score > 0 else 0

    return {
        "CHROM": contig,
        "POS": pos,
        "REF": row["REF"],
        "ALT": row["ALT"],
        "QUAL": qual,
        "GT": gt,
        "Databases": databases,
        "Gene Name": [row["GENE_SYMBOL"]] if row.get("GENE_SYMBOL") else [],
        "1KG Frequency": kg_freq,
        "ClinVar SIG": clinvar_sig,
        "DP": None,
        "AD": [-1, -1],
        "gnomAD AF": gnomad_af,
        "gnomADg AF": gnomad_genome_af,
        "ACMG": acmg,
        "OMIM": omim,
        "Libra Pathogenicity": None,
        "Sorting Score": sorting_score,
        "ACMG Severity": acmg_severity,
        "Pathogenicity ClinVar Severity": clinvar_severity,
        "Alpha Missense Score": None,
        "Alpha Missense Prediction": None,
        "SIFT": sift,
        "Polyphen": polyphen,
        "CADD": cadd,
        "REVEL": revel,
        "SpliceAI": None,
        "DANN": None,
        "MetalR": None,
        "Gene Phenotype": gene_phenotype,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(tsv_path: Path, genes_tsv_path: Path, sample_name: str,
            top_genes: int, output_path: Path) -> int:
    # 1. Read gene ranks from genes TSV to identify top-N genes
    top_gene_set = set()
    with open(genes_tsv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            # columns: #RANK ID GENE_SYMBOL ...
            try:
                rank = int(parts[0])
                gene_symbol = parts[2]
            except (IndexError, ValueError):
                continue
            if rank <= top_genes:
                top_gene_set.add(gene_symbol)

    print(f"Top {top_genes} genes: {len(top_gene_set)} unique gene symbols")

    # 2. Read variant TSV and filter to top genes
    records = []
    headers = None
    with open(tsv_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                headers = line.lstrip("#").split("\t")
                continue
            if headers is None:
                continue
            row = dict(zip(headers, line.split("\t")))
            if row.get("GENE_SYMBOL") not in top_gene_set:
                continue

            try:
                gene_rank = int(row.get("RANK", 0))
            except (ValueError, TypeError):
                gene_rank = 0

            records.append(row_to_record(row, gene_rank))

    # 3. Sort by Sorting Score descending (preserves Exomiser rank order)
    records.sort(key=lambda r: r["Sorting Score"], reverse=True)

    # 4. Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} variants to {output_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Convert Exomiser TSV_VARIANT to DAVP JSONL")
    parser.add_argument("--tsv", required=True, type=Path, help="Exomiser variants TSV path")
    parser.add_argument("--genes", required=True, type=Path, help="Exomiser genes TSV path")
    parser.add_argument("--sample", required=True, help="Sample name (used for output filename)")
    parser.add_argument("--top-genes", type=int, default=256, help="Keep variants in top N genes (default: 256)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    output_path = args.output_dir / f"{args.sample}.jsonl"
    convert(args.tsv, args.genes, args.sample, args.top_genes, output_path)


if __name__ == "__main__":
    main()
