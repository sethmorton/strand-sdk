import gzip
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from campaigns.abca4.src.data.filter_abca4_variants import ABCA4VariantFilter
from campaigns.abca4.src.annotation.annotate_transcripts import VariantAnnotator


def test_clinsig_normalization_handles_mixed_delimiters(tmp_path: Path):
    data_root = tmp_path
    clinvar_dir = data_root / "clinvar"
    clinvar_dir.mkdir(parents=True)

    rows = [
        {
            "Chromosome": "1",
            "Start": 94400001,
            "Stop": 94400002,
            "ReferenceAllele": "A",
            "AlternateAllele": "G",
            "GeneSymbol": "ABCA4",
            "ClinicalSignificance": "Uncertain significance|Pathogenic",
            "ReviewStatus": "criteria provided, single submitter",
            "RS# (dbSNP)": "rs1",
            "PhenotypeList": "",
            "Origin": "germline",
            "Assembly": "GRCh38",
        },
        {
            "Chromosome": "1",
            "Start": 94400003,
            "Stop": 94400004,
            "ReferenceAllele": "T",
            "AlternateAllele": "C",
            "GeneSymbol": "ABCA4",
            "ClinicalSignificance": "Benign",
            "ReviewStatus": "criteria provided, single submitter",
            "RS# (dbSNP)": "rs2",
            "PhenotypeList": "",
            "Origin": "germline",
            "Assembly": "GRCh38",
        },
    ]
    df = pd.DataFrame(rows)
    gz_path = clinvar_dir / "variant_summary.txt.gz"
    with gzip.open(gz_path, "wt") as handle:
        df.to_csv(handle, sep="\t", index=False)

    flt = ABCA4VariantFilter(input_dir=data_root, output_dir=tmp_path / "processed")
    clinvar_df = flt.load_clinvar_tsv()
    filtered = flt.filter_abca4_variants(clinvar_df)

    assert len(filtered) == 1
    assert filtered.iloc[0]["ref"] == "A"
    assert filtered.iloc[0]["clinical_significance"] == "uncertain significance"


def test_compute_structural_context_within_and_outside_exons():
    annotator = VariantAnnotator.__new__(VariantAnnotator)  # bypass __init__
    exon_a = SimpleNamespace(start=10, end=20)
    exon_b = SimpleNamespace(start=40, end=50)
    annotator.transcript = SimpleNamespace(start=1, end=100, exons=[exon_a, exon_b])

    exonic = annotator._compute_structural_context(15)
    assert exonic["genomic_region"] == "exonic"
    assert exonic["coding_impact"] == "coding_snv"
    assert exonic["intron_distance"] == 5

    upstream = annotator._compute_structural_context(-10)
    assert upstream["genomic_region"] == "upstream"
    assert upstream["coding_impact"] == "upstream"
    assert upstream["intron_distance"] == 11
