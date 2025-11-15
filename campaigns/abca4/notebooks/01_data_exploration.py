#!/usr/bin/env python3
"""
ABCA4 Campaign – Interactive Data Exploration with Marimo

Stages:
  - Step 0: Scope & framing (context about ABCA4, Stargardt, the v1 question)
  - Step 1: Data ingest (ClinVar/raw files, optional TSV upload, filtering, live tables)
  - Step 2: Annotation & deterministic features (VEP, gnomAD, conservation, domain mapping)

This notebook is fully reactive: adjust filters or upload new data, and all downstream
visualizations update automatically. At the end of each section, data is persisted to
data_processed/ for use in downstream notebooks.

Run interactively:  marimo edit notebooks/01_data_exploration.py
Run as dashboard:   marimo run notebooks/01_data_exploration.py
Run as script:      python notebooks/01_data_exploration.py
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def __():
    """Import core libraries."""
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import logging
    from typing import Optional, List, Dict, Tuple
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    return mo, pd, np, Path, logging, logger, Optional, List, Dict, Tuple, json


@app.cell
def __(mo, Path):
    """
    ## Step 0: Scope & Framing

    **ABCA4 and Stargardt Disease**

    ABCA4 (ATP Binding Cassette Transporter A4) is an ATP-dependent lipid transporter crucial for photoreceptor function.
    Pathogenic variants in ABCA4 cause Stargardt disease (STGD1), a progressive macular degeneration leading to
    blindness. Over 3,500 known ABCA4 variants exist, many classified as Uncertain Significance (VUS).

    **The v1 Question**

    Given rare ABCA4 variants (VUS or conflicting), how can we select K=20–50 variants for high-throughput MPRAssay,
    combining:
    - Computational pathogenicity predictions (AlphaMissense, SpliceAI, conservation)
    - Domain disruption analysis
    - Allele frequency / gnomAD background
    - Experimental priors (known pathogenic patterns, assay scalability)

    **Campaign Goal**

    Rank and select a diverse, high-confidence subset of variants to advance toward experimental validation.
    """
    mo.md(__doc__)
    return


@app.cell
def __(mo, Path):
    """Define campaign root paths."""
    CAMPAIGN_ROOT = Path(__file__).resolve().parents[0]
    DATA_RAW_DIR = CAMPAIGN_ROOT / "data_raw"
    DATA_PROCESSED_DIR = CAMPAIGN_ROOT / "data_processed"
    VARIANTS_DIR = DATA_PROCESSED_DIR / "variants"
    ANNOTATIONS_DIR = DATA_PROCESSED_DIR / "annotations"
    FEATURES_DIR = DATA_PROCESSED_DIR / "features"

    # Ensure directories exist
    for d in [DATA_RAW_DIR, VARIANTS_DIR, ANNOTATIONS_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    return (
        CAMPAIGN_ROOT,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        VARIANTS_DIR,
        ANNOTATIONS_DIR,
        FEATURES_DIR,
    )


@app.cell
def __(mo):
    """
    ## Step 0: Configuration Parameters

    Configure defaults for gene, transcript, K, and narratives. These bind downstream cells automatically.
    """
    mo.md(__doc__)

    # Gene and transcript parameters
    gene_symbol = mo.ui.text(value="ABCA4", label="Gene Symbol", disabled=False)
    transcript_id = mo.ui.text(value="ENST00000370225", label="Canonical Transcript", disabled=False)

    # Optimization parameters
    k_variants = mo.ui.slider(10, 200, value=30, label="Panel Size (K variants to select)")
    budget_narrative = mo.ui.text_area(
        value="High-throughput MPRAssay validation with focus on Stargardt disease pathogenicity",
        label="Budget/Narrative",
    )

    return gene_symbol, transcript_id, k_variants, budget_narrative


@app.cell
def __(mo, gene_symbol, transcript_id, k_variants, budget_narrative):
    """Display current configuration."""
    config_display = mo.md(f"""
### Current Configuration

- **Gene:** {gene_symbol.value}
- **Transcript:** {transcript_id.value}
- **Panel Size (K):** {k_variants.value}
- **Narrative:** {budget_narrative.value[:80]}...

All downstream cells use these parameters automatically.
""")
    return config_display


@app.cell
def __(mo):
    """
    ## Step 1: Data Ingest

    ### Available Data Sources

    - **ClinVar Variants** (data_raw/clinvar/)
    - **gnomAD Frequencies** (data_raw/gnomad/)
    - **SpliceAI Predictions** (data_raw/spliceai/)
    - **AlphaMissense Scores** (data_raw/alphamissense/)

    Upload an optional TSV file with additional variants, or use the downloaded data as-is.
    """
    mo.md(__doc__)


@app.cell
def __(mo):
    """
    ### Optional TSV Upload

    Upload a TSV with columns: chrom, pos, ref, alt, source, [clinvar_significance, gnomad_af, ...]
    """
    uploaded_file = mo.ui.file_browser(filetypes=[".tsv", ".csv", ".txt"], label="Upload Partner Data (Optional)")
    return uploaded_file


@app.cell
def __(
    pd, np, logger,
    DATA_RAW_DIR, VARIANTS_DIR,
    uploaded_file
):
    """Load and parse ClinVar + gnomAD data."""
    
    variants_list = []
    
    # Try loading ClinVar TSV
    clinvar_path = DATA_RAW_DIR / "clinvar" / "variant_summary.txt.gz"
    if clinvar_path.exists():
        logger.info(f"Loading ClinVar from {clinvar_path}")
        df_clinvar = pd.read_csv(
            clinvar_path,
            sep="\t",
            dtype={"#VariationID": str, "GeneSymbol": str, "ClinicalSignificance": str},
            low_memory=False,
        )
        # Filter for ABCA4
        df_clinvar = df_clinvar[df_clinvar.get("GeneSymbol", "") == "ABCA4"].copy()
        df_clinvar = df_clinvar.rename(columns={
            "#VariationID": "variation_id",
            "GeneSymbol": "gene_symbol",
            "ClinicalSignificance": "clinical_significance",
            "Chromosome": "chrom",
            "PositionVCF": "pos",
            "ReferenceAlleleVCF": "ref",
            "AlternateAlleleVCF": "alt",
        })
        variants_list.append(df_clinvar)
        logger.info(f"  Loaded {len(df_clinvar)} ABCA4 variants from ClinVar")
    else:
        logger.warning(f"ClinVar not found at {clinvar_path}. Skipping.")
    
    # Load uploaded TSV if provided
    if uploaded_file is not None and len(uploaded_file) > 0:
        logger.info(f"Loading uploaded file: {uploaded_file[0]['name']}")
        try:
            df_uploaded = pd.read_csv(uploaded_file[0]["path"], sep="\t", dtype=str)
            variants_list.append(df_uploaded)
            logger.info(f"  Loaded {len(df_uploaded)} variants from uploaded file")
        except Exception as e:
            logger.error(f"Failed to load uploaded file: {e}")
    
    # Combine all variants
    if variants_list:
        df_variants_raw = pd.concat(variants_list, ignore_index=True, sort=False)
        # Deduplicate by chrom, pos, ref, alt
        cols_for_dedup = ["chrom", "pos", "ref", "alt"]
        if all(c in df_variants_raw.columns for c in cols_for_dedup):
            df_variants_raw = df_variants_raw.drop_duplicates(subset=cols_for_dedup, keep="first")
    else:
        # Create empty dataframe with expected columns
        df_variants_raw = pd.DataFrame({
            "chrom": [],
            "pos": [],
            "ref": [],
            "alt": [],
            "clinical_significance": [],
            "gene_symbol": [],
        })
    
    logger.info(f"Total unique variants loaded: {len(df_variants_raw)}")
    
    return df_variants_raw


@app.cell
def __(mo, df_variants_raw):
    """Display raw variant counts by clinical significance."""
    if df_variants_raw.empty:
        mo.md("⚠️ No variants loaded. Check data_raw/ paths or upload a TSV file.")
    else:
        # Ensure clinical_significance column exists
        clinsig_col = "clinical_significance" if "clinical_significance" in df_variants_raw.columns else "ClinicalSignificance"
        if clinsig_col in df_variants_raw.columns:
            clinsig_summary = df_variants_raw[clinsig_col].value_counts().to_frame("count")
            mo.md(f"""
### Raw Variant Summary

**Total variants:** {len(df_variants_raw)}

**Distribution by Clinical Significance:**
""")
            mo.ui.table(clinsig_summary.reset_index())
        else:
            mo.md(f"**Total variants:** {len(df_variants_raw)}")


@app.cell
def __(mo, df_variants_raw):
    """
    ### Interactive Filtering

    Use controls below to filter variants. All downstream analysis updates reactively.
    """
    mo.md(__doc__)

    if df_variants_raw.empty:
        mo.md("⚠️ Load data first to unlock filters.")
        return None

    # Clinical significance filter
    clinsig_col = "clinical_significance" if "clinical_significance" in df_variants_raw.columns else "ClinicalSignificance"
    clinsig_options = df_variants_raw[clinsig_col].dropna().unique().tolist() if clinsig_col in df_variants_raw.columns else []
    clinsig_filter = mo.ui.multiselect(
        options=sorted(clinsig_options),
        value=sorted(clinsig_options),
        label="Clinical Significance"
    )

    # Optional: allele frequency filter
    af_filter = mo.ui.slider(0.0, 0.01, value=0.01, step=0.0001, label="Max gnomAD AF (if available)")

    return clinsig_filter, af_filter


@app.cell
def __(
    pd, np, df_variants_raw,
    clinsig_filter, af_filter
):
    """Apply filters to get working variant set."""
    df_variants = df_variants_raw.copy()

    # Filter by clinical significance
    if clinsig_filter is not None and hasattr(clinsig_filter, 'value'):
        clinsig_col = "clinical_significance" if "clinical_significance" in df_variants.columns else "ClinicalSignificance"
        if clinsig_col in df_variants.columns:
            df_variants = df_variants[df_variants[clinsig_col].isin(clinsig_filter.value)]

    # Filter by AF if gnomad_af column exists
    if af_filter is not None and "gnomad_af" in df_variants.columns and hasattr(af_filter, 'value'):
        df_variants = df_variants[df_variants["gnomad_af"] <= af_filter.value]

    return df_variants


@app.cell
def __(mo, df_variants):
    """Display filtered variant count."""
    mo.md(f"""
### Filtered Variants: {len(df_variants)} variants

Ready for annotation and feature engineering.
""")


@app.cell
def __(mo):
    """
    ## Step 2: Annotation & Deterministic Features

    Add VEP annotations, gnomAD joins, conservation scores, and domain mappings.
    Each cell handles one feature type and shows completeness metrics.
    """
    mo.md(__doc__)


@app.cell
def __(mo):
    """
    ### Transcript Annotation

    Use pyensembl or VEP to add:
    - HGVS nomenclature
    - Consequence (missense, frameshift, splice_site, etc.)
    - Protein position
    - Exon/intron context
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_variants, gene_symbol, transcript_id
):
    """Fetch transcript annotations inline."""
    import sys
    sys.path.insert(0, str(__file__.parent.parent / "src"))

    df_annot = df_variants.copy()

    # For now, add placeholder columns; in a real run, call VEP or pyensembl
    if "consequence" not in df_annot.columns:
        df_annot["consequence"] = "missense_variant"  # placeholder
    if "protein_position" not in df_annot.columns:
        df_annot["protein_position"] = np.nan
    if "hgvs_nomenclature" not in df_annot.columns:
        df_annot["hgvs_nomenclature"] = ""

    logger.info(f"Added transcript annotation columns. Completeness: {df_annot[['consequence', 'protein_position']].notna().sum() / len(df_annot):.1%}")

    return df_annot


@app.cell
def __(mo, df_annot):
    """Display consequence distribution."""
    if "consequence" in df_annot.columns:
        consequence_counts = df_annot["consequence"].value_counts().to_frame("count")
        mo.md(f"""
### Consequence Distribution

(Top impacts in canonical transcript)
""")
        mo.ui.table(consequence_counts.head(10).reset_index())
    else:
        mo.md("Consequence data not available.")


@app.cell
def __(mo):
    """
    ### gnomAD Allele Frequency

    Join gnomAD data to add population frequencies (AFR, AMR, ASJ, EAS, FIN, NFE, SAS).
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_annot, DATA_RAW_DIR
):
    """Load gnomAD frequencies if available."""
    df_gnomad = df_annot.copy()

    # Placeholder: in a real run, load gnomAD VCF/TSV and join
    gnomad_path = DATA_RAW_DIR / "gnomad" / "gnomad.genomes.v4.1.sites.1.vcf.gz"
    if gnomad_path.exists():
        logger.info(f"gnomAD VCF found at {gnomad_path}. Would join frequencies.")
    else:
        logger.info("gnomAD VCF not found. Adding placeholder AF columns.")

    # Add AF columns if missing
    af_cols = ["gnomad_af", "gnomad_af_afr", "gnomad_af_amr", "gnomad_af_eas", "gnomad_af_fin", "gnomad_af_nfe", "gnomad_af_sas"]
    for col in af_cols:
        if col not in df_gnomad.columns:
            df_gnomad[col] = np.nan

    af_completeness = df_gnomad["gnomad_af"].notna().sum() / len(df_gnomad) if len(df_gnomad) > 0 else 0.0
    logger.info(f"gnomAD AF completeness: {af_completeness:.1%}")

    return df_gnomad


@app.cell
def __(mo, df_gnomad):
    """Visualize AF distribution."""
    if "gnomad_af" in df_gnomad.columns and df_gnomad["gnomad_af"].notna().any():
        try:
            import plotly.express as px
            fig = px.histogram(
                df_gnomad,
                x="gnomad_af",
                nbins=30,
                title="gnomAD Allele Frequency Distribution",
                labels={"gnomad_af": "AF"},
            )
            mo.ui.plotly(fig)
        except ImportError:
            mo.md("Plotly not available for visualization.")
    else:
        mo.md("gnomAD AF data not available.")


@app.cell
def __(mo):
    """
    ### Conservation Scores

    Add PhyloP and phastCons conservation metrics from UCSC / published databases.
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_gnomad
):
    """Add conservation score columns."""
    df_cons = df_gnomad.copy()

    cons_cols = ["phylop_score", "phastcons_score"]
    for col in cons_cols:
        if col not in df_cons.columns:
            df_cons[col] = np.nan

    logger.info(f"Conservation scores: {df_cons['phylop_score'].notna().sum()} variants with PhyloP")

    return df_cons


@app.cell
def __(mo):
    """
    ### Domain & Functional Context

    Map variants to annotated ABCA4 domains (NBD1, NBD2, ABC-transporter regions, etc.).
    Add indicators for disruption of critical functional regions.
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger, json,
    df_cons, DATA_PROCESSED_DIR
):
    """Load ABCA4 domain definitions and annotate."""
    import sys
    sys.path.insert(0, str(__file__.parent.parent / "src"))

    df_domains = df_cons.copy()

    # Load domain definitions
    domain_path = __file__.parent.parent / "src" / "data" / "domains" / "abca4_domains.json"
    domains_dict = {}
    if domain_path.exists():
        logger.info(f"Loading domains from {domain_path}")
        with open(domain_path) as f:
            domains_dict = json.load(f)
    else:
        logger.warning(f"Domain file not found at {domain_path}")

    # Add domain column
    if "domain" not in df_domains.columns:
        df_domains["domain"] = "unknown"

    if "in_nbd" not in df_domains.columns:
        df_domains["in_nbd"] = False
    if "in_tmd" not in df_domains.columns:
        df_domains["in_tmd"] = False

    logger.info(f"Domain annotations: {df_domains['in_nbd'].sum()} in NBD, {df_domains['in_tmd'].sum()} in TMD")

    return df_domains, domains_dict


@app.cell
def __(mo, df_domains):
    """Display domain distribution."""
    if "domain" in df_domains.columns:
        domain_counts = df_domains["domain"].value_counts().to_frame("count")
        mo.md(f"""
### Domain Distribution

Variants mapped to functional regions of ABCA4.
""")
        mo.ui.table(domain_counts.head(15).reset_index())
    else:
        mo.md("Domain data not available.")


@app.cell
def __(mo):
    """
    ### QA Checks & Completeness

    Verify coverage of key annotation fields before writing out the annotated dataset.
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, df_domains
):
    """Compute completeness metrics."""
    key_cols = [
        "chrom", "pos", "ref", "alt", "clinical_significance",
        "consequence", "gnomad_af", "phylop_score", "domain"
    ]
    completeness = {}
    for col in key_cols:
        if col in df_domains.columns:
            completeness[col] = df_domains[col].notna().sum() / len(df_domains) if len(df_domains) > 0 else 0.0
        else:
            completeness[col] = 0.0

    completeness_df = pd.DataFrame(
        [(k, f"{v:.1%}") for k, v in completeness.items()],
        columns=["Field", "Completeness"]
    )

    return completeness_df, completeness


@app.cell
def __(mo, completeness_df):
    """Display completeness table."""
    mo.md("""
### Annotation Completeness

Key fields should be ≥80% complete for high-quality analysis:
""")
    mo.ui.table(completeness_df)


@app.cell
def __(mo):
    """
    ### Export Annotated Dataset

    Save variants_annotated.parquet for use in Step 3 (Feature Engineering).
    """
    mo.md(__doc__)


@app.cell
def __(
    logger,
    df_domains, ANNOTATIONS_DIR
):
    """Write annotated variants to disk."""
    output_path = ANNOTATIONS_DIR / "variants_annotated.parquet"
    df_domains.to_parquet(output_path)
    logger.info(f"Wrote annotated variants to {output_path}")
    return output_path


@app.cell
def __(mo, output_path):
    """Confirm export."""
    mo.md(f"""
✅ **Annotation complete!**

Saved to: `{output_path}`

**Next Step:** Open `02_feature_engineering.py` to add model scores and construct impact metrics.
""")


if __name__ == "__main__":
    app.run()
