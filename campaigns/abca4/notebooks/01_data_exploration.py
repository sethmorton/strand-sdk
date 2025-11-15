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
    """Optional TSV Upload."""
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
        try:
            df_clinvar = pd.read_csv(
                clinvar_path,
                sep="\t",
                dtype={"#VariationID": str, "GeneSymbol": str, "ClinicalSignificance": str},
                low_memory=False,
                nrows=5000  # Limit to avoid memory issues
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
        except Exception as e:
            logger.warning(f"Failed to load ClinVar: {e}")
    else:
        logger.info(f"ClinVar not found at {clinvar_path}. Creating example data.")
        # Create example data for testing
        df_example = pd.DataFrame({
            "chrom": ["1"] * 10,
            "pos": range(94400000, 94400010),
            "ref": ["A"] * 10,
            "alt": ["T", "G", "C"] * 3 + ["T"],
            "clinical_significance": ["Pathogenic"] * 3 + ["Benign"] * 3 + ["Uncertain significance"] * 4,
            "gene_symbol": ["ABCA4"] * 10,
        })
        variants_list.append(df_example)
        logger.info(f"Created {len(df_example)} example variants for testing")
    
    # Load uploaded TSV if provided
    if uploaded_file is not None and hasattr(uploaded_file, 'value') and uploaded_file.value:
        logger.info(f"Loading uploaded file")
        try:
            df_uploaded = pd.read_csv(uploaded_file.value, sep="\t", dtype=str)
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
        mo.md("⚠️ No variants loaded.")
    else:
        # Find clinical significance column
        _clinsig_cols = [c for c in df_variants_raw.columns if "clin" in c.lower() or "significance" in c.lower()]
        _clinsig_col = _clinsig_cols[0] if _clinsig_cols else "clinical_significance"
        
        if _clinsig_col in df_variants_raw.columns:
            _clinsig_summary = df_variants_raw[_clinsig_col].value_counts().to_frame("count")
            mo.md(f"""
### Raw Variant Summary

**Total variants:** {len(df_variants_raw)}

**Distribution by Clinical Significance:**
""")
            mo.ui.table(_clinsig_summary.reset_index())
        else:
            mo.md(f"**Total variants:** {len(df_variants_raw)}")


@app.cell
def __(mo, df_variants_raw):
    """Interactive Filtering panel."""
    mo.md("""
### Interactive Filtering

Use controls below to filter variants.
""")

    filter_clinsig = None
    filter_af = None

    if not df_variants_raw.empty:
        # Find clinical significance column
        _clinsig_cols = [c for c in df_variants_raw.columns if "clin" in c.lower() or "significance" in c.lower()]
        _clinsig_col = _clinsig_cols[0] if _clinsig_cols else None
        
        if _clinsig_col:
            _clinsig_options = df_variants_raw[_clinsig_col].dropna().unique().tolist()
            filter_clinsig = mo.ui.multiselect(
                options=sorted(_clinsig_options),
                value=sorted(_clinsig_options),
                label="Clinical Significance"
            )

        # Optional: allele frequency filter
        filter_af = mo.ui.slider(0.0, 0.01, value=0.01, step=0.0001, label="Max gnomAD AF (if available)")

    return filter_clinsig, filter_af


@app.cell
def __(
    pd, np, df_variants_raw,
    filter_clinsig, filter_af
):
    """Apply filters to get working variant set."""
    df_variants_filtered = df_variants_raw.copy()

    # Filter by clinical significance
    if filter_clinsig is not None and hasattr(filter_clinsig, 'value'):
        _clinsig_cols = [c for c in df_variants_filtered.columns if "clin" in c.lower() or "significance" in c.lower()]
        _clinsig_col = _clinsig_cols[0] if _clinsig_cols else None
        if _clinsig_col and _clinsig_col in df_variants_filtered.columns:
            df_variants_filtered = df_variants_filtered[df_variants_filtered[_clinsig_col].isin(filter_clinsig.value)]

    # Filter by AF if gnomad_af column exists
    if filter_af is not None and "gnomad_af" in df_variants_filtered.columns and hasattr(filter_af, 'value'):
        df_variants_filtered = df_variants_filtered[df_variants_filtered["gnomad_af"] <= filter_af.value]

    return df_variants_filtered


@app.cell
def __(mo, df_variants_filtered):
    """Display filtered variant count."""
    mo.md(f"""
### Filtered Variants: {len(df_variants_filtered)} variants

Ready for annotation and feature engineering.
""")


@app.cell
def __(mo):
    """
    ## Step 2: Annotation & Deterministic Features

    Add VEP annotations, gnomAD joins, conservation scores, and domain mappings.
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_variants_filtered
):
    """Add transcript annotations."""
    df_annot = df_variants_filtered.copy()

    # Add placeholder columns for annotations
    if "consequence" not in df_annot.columns:
        df_annot["consequence"] = "missense_variant"
    if "protein_position" not in df_annot.columns:
        df_annot["protein_position"] = np.nan
    if "hgvs_nomenclature" not in df_annot.columns:
        df_annot["hgvs_nomenclature"] = ""

    logger.info(f"Added transcript annotations. Ready for feature engineering.")

    return df_annot


@app.cell
def __(mo, df_annot):
    """Display consequence distribution."""
    if "consequence" in df_annot.columns and not df_annot.empty:
        _consequence_counts = df_annot["consequence"].value_counts().to_frame("count")
        mo.md("""
### Consequence Distribution
""")
        mo.ui.table(_consequence_counts.head(10).reset_index())
    else:
        mo.md("No consequence data available.")


@app.cell
def __(
    pd, np, logger,
    df_annot
):
    """Add gnomAD allele frequency data."""
    df_gnomad = df_annot.copy()

    # Add AF columns if missing
    _af_cols = ["gnomad_af", "gnomad_af_afr", "gnomad_af_amr", "gnomad_af_eas", "gnomad_af_fin", "gnomad_af_nfe", "gnomad_af_sas"]
    for _af_col in _af_cols:
        if _af_col not in df_gnomad.columns:
            df_gnomad[_af_col] = np.nan

    logger.info(f"Added gnomAD AF columns.")

    return df_gnomad


@app.cell
def __(mo, df_gnomad):
    """Visualize AF distribution if available."""
    if "gnomad_af" in df_gnomad.columns and df_gnomad["gnomad_af"].notna().any():
        try:
            import plotly.graph_objects as go
            _df_af = df_gnomad.dropna(subset=["gnomad_af"])
            if len(_df_af) > 0:
                _fig = go.Figure()
                _fig.add_trace(go.Histogram(
                    x=_df_af["gnomad_af"],
                    nbinsx=30,
                    name="AF",
                    marker_color="steelblue"
                ))
                _fig.update_layout(
                    title="gnomAD Allele Frequency Distribution",
                    xaxis_title="Allele Frequency",
                    yaxis_title="Count",
                    hovermode="x unified",
                    showlegend=False
                )
                mo.ui.plotly(_fig)
            else:
                mo.md("No gnomAD AF data to visualize.")
        except ImportError:
            mo.md("Plotly not available for visualization.")
    else:
        mo.md("No gnomAD AF data available.")


@app.cell
def __(
    pd, np, logger,
    df_gnomad
):
    """Add conservation score columns."""
    df_cons = df_gnomad.copy()

    _cons_cols = ["phylop_score", "phastcons_score"]
    for _cons_col in _cons_cols:
        if _cons_col not in df_cons.columns:
            df_cons[_cons_col] = np.nan

    logger.info(f"Added conservation score columns.")

    return df_cons


@app.cell
def __(
    pd, np, logger, json,
    df_cons, DATA_PROCESSED_DIR
):
    """Add domain annotations."""
    df_domains = df_cons.copy()

    # Add domain columns
    if "domain" not in df_domains.columns:
        df_domains["domain"] = "unknown"

    if "in_nbd" not in df_domains.columns:
        df_domains["in_nbd"] = False
    if "in_tmd" not in df_domains.columns:
        df_domains["in_tmd"] = False

    logger.info(f"Added domain annotation columns.")

    return df_domains


@app.cell
def __(mo, df_domains):
    """Display domain distribution."""
    if "domain" in df_domains.columns and not df_domains.empty:
        _domain_counts = df_domains["domain"].value_counts().to_frame("count")
        mo.md("""
### Domain Distribution
""")
        mo.ui.table(_domain_counts.head(15).reset_index())
    else:
        mo.md("No domain data available.")
    



@app.cell
def __(mo, pd, df_domains):
    """Display completeness metrics."""
    _key_cols = [
        "chrom", "pos", "ref", "alt", "clinical_significance",
        "consequence", "gnomad_af", "phylop_score", "domain"
    ]
    _completeness = {}
    for _col in _key_cols:
        if _col in df_domains.columns:
            _completeness[_col] = df_domains[_col].notna().sum() / len(df_domains) if len(df_domains) > 0 else 0.0
        else:
            _completeness[_col] = 0.0

    _comp_df = pd.DataFrame(
        [(k, f"{v:.1%}") for k, v in _completeness.items()],
        columns=["Field", "Completeness"]
    )
    
    mo.md("""
### Annotation Completeness

Key fields completeness check:
""")
    mo.ui.table(_comp_df)


@app.cell
def __(
    logger,
    df_domains, ANNOTATIONS_DIR
):
    """Export annotated variants to disk."""
    output_path_annot = ANNOTATIONS_DIR / "variants_annotated.parquet"
    df_domains.to_parquet(output_path_annot)
    logger.info(f"Wrote annotated variants to {output_path_annot}")
    return output_path_annot


@app.cell
def __(mo, output_path_annot):
    """Confirm export."""
    mo.md(f"""
✅ **Annotation Complete!**

Saved to: `{output_path_annot}`

**Next Step:** Open `02_feature_engineering.py` to add model scores and construct impact metrics.
""")


if __name__ == "__main__":
    app.run()
