#!/usr/bin/env python3
"""ABCA4 Campaign – interactive data exploration with Marimo."""

import marimo

__generated_with = "0.17.8"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    return mo, np, pd, Path

@app.cell
def __(Path):
    CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
    FEATURE_MATRIX = CAMPAIGN_ROOT / "data_processed" / "features" / "abca4_feature_matrix.parquet"
    return CAMPAIGN_ROOT, FEATURE_MATRIX

@app.cell
def __(mo):
    mo.md("# 🔬 ABCA4 Variant Explorer - Marimo Notebook")
    mo.md(
        """
    Use the controls below to slice the unified ABCA4 feature matrix. Every widget is
    reactive: once the data is filtered, tables, charts, and download buttons update
    automatically.
    """
    )
    return

@app.cell
def __(FEATURE_MATRIX, np, pd):
    if not FEATURE_MATRIX.exists():
        print(f"⚠️ Missing feature matrix at {FEATURE_MATRIX}. Run `invoke features.compute`. ")
        df_features = pd.DataFrame()
    else:
        df_features = pd.read_parquet(FEATURE_MATRIX)
        df_features = df_features.replace({np.inf: np.nan, -np.inf: np.nan})
    return df_features,

@app.cell
def __(pd):
    def _safe_unique(series: pd.Series) -> list[str]:
        if series.empty:
            return []
        return sorted(series.dropna().astype(str).unique())
    return _safe_unique,

@app.cell
def __(df_features, _safe_unique, mo, pd):
    if df_features.empty:
        filter_panel = mo.md("Upload data to unlock controls.")
        clinical_significance = None
        coding_impacts = None
        regulatory_types = None
        domain_only = None
        allele_frequency_max = None
        splice_threshold = None
        missense_threshold = None
    else:
        clinical_significance = mo.ui.multiselect(
            options=_safe_unique(df_features["clinical_significance"]),
            value=_safe_unique(df_features["clinical_significance"]),
            label="Clinical Significance",
        )

        coding_impacts = mo.ui.multiselect(
            options=_safe_unique(df_features["coding_impact"]),
            value=_safe_unique(df_features["coding_impact"]),
            label="Coding Impact",
        )

        regulatory_types = mo.ui.multiselect(
            options=_safe_unique(df_features["regulatory_type"]),
            value=_safe_unique(df_features["regulatory_type"]),
            label="Regulatory Context",
        )

        domain_only = mo.ui.switch(False, label="Only keep variants inside annotated domains")

        max_af = float(df_features.get("gnomad_max_af", pd.Series([0.0])).fillna(0).max() or 0.01)
        allele_frequency_max = mo.ui.slider(0.0, max(0.01, max_af), value=min(0.01, max_af), step=0.0001, label="Max gnomAD AF")

        splice_threshold = mo.ui.slider(0.0, 1.0, value=0.1, step=0.01, label="Min SpliceAI score")
        missense_threshold = mo.ui.slider(0.0, 1.0, value=0.3, step=0.01, label="Min missense score")

        filter_panel = mo.md(
            """
### Filters

**Clinical Significance:**\n{} **Coding Impact:**\n{}\n\n**Regulatory Context:**\n{} **Domain Filter:**\n{}\n\n**gnomAD AF:**\n{} **SpliceAI:**\n{} **Missense:**\n{}
""".format(
                clinical_significance, coding_impacts,
                regulatory_types, domain_only,
                allele_frequency_max, splice_threshold, missense_threshold
            )
        )
    
    return (
        allele_frequency_max,
        clinical_significance,
        coding_impacts,
        coding_impacts,
        domain_only,
        filter_panel,
        missense_threshold,
        regulatory_types,
        splice_threshold,
    )

@app.cell
def __(mo):
    reg_weight = mo.ui.slider(0.0, 1.0, value=0.35, step=0.05, label="Regulatory weight")
    splice_weight = mo.ui.slider(0.0, 1.0, value=0.25, step=0.05, label="Splice weight")
    missense_weight = mo.ui.slider(0.0, 1.0, value=0.25, step=0.05, label="Missense weight")
    conservation_weight = mo.ui.slider(0.0, 1.0, value=0.15, step=0.05, label="Conservation weight")
    
    weights_content = mo.md("Tune the scoring weights (will normalize automatically).")
    
    return reg_weight, splice_weight, missense_weight, conservation_weight, weights_content

@app.cell
def __(mo):
    top_k_slider = mo.ui.slider(5, 200, value=50, step=5, label="Top-K for downloads")
    return top_k_slider,

@app.cell
def __(conservation_weight, missense_weight, np, reg_weight, splice_weight):
    sliders = [reg_weight, splice_weight, missense_weight, conservation_weight]
    values = np.array([slider.value if slider else 0.25 for slider in sliders], dtype=float)
    if values.sum() == 0:
        values = np.ones_like(values)
    values = values / values.sum()
    score_weights = {
        "regulatory_score": values[0],
        "spliceai_max_score": values[1],
        "missense_combined_score": values[2],
        "conservation_score": values[3],
    }
    return score_weights,

@app.cell
def __(
    allele_frequency_max,
    clinical_significance,
    coding_impacts,
    df_features,
    domain_only,
    missense_threshold,
    regulatory_types,
    score_weights,
    splice_threshold,
):
    current_variants = df_features.copy()
    
    if not current_variants.empty:
        if clinical_significance and clinical_significance.value:
            current_variants = current_variants[current_variants["clinical_significance"].isin(clinical_significance.value)]

        if coding_impacts and coding_impacts.value:
            current_variants = current_variants[current_variants["coding_impact"].isin(coding_impacts.value)]

        if regulatory_types and regulatory_types.value:
            current_variants = current_variants[current_variants["regulatory_type"].isin(regulatory_types.value)]

        if domain_only and domain_only.value:
            current_variants = current_variants[current_variants.get("in_domain", 0) == 1]

        if allele_frequency_max:
            current_variants = current_variants[current_variants.get("gnomad_max_af", 0).fillna(0) <= allele_frequency_max.value]

        if splice_threshold:
            current_variants = current_variants[current_variants.get("spliceai_max_score", 0).fillna(0) >= splice_threshold.value]

        if missense_threshold:
            current_variants = current_variants[current_variants.get("missense_combined_score", 0).fillna(0) >= missense_threshold.value]

        for feature, weight in score_weights.items():
            if feature in current_variants.columns:
                current_variants[feature] = current_variants[feature].fillna(0)

        if score_weights:
            current_variants = current_variants.assign(
                composite_score=sum(weight * current_variants.get(feature, 0) for feature, weight in score_weights.items())
            )
            current_variants = current_variants.sort_values("composite_score", ascending=False)
        
        current_variants = current_variants.reset_index(drop=True)
    
    return current_variants,

@app.cell
def __(current_variants, df_features, mo, np, pd):
    if df_features.empty:
        summary_output = mo.md("No data loaded.")
    else:
        total = len(current_variants)
        inside_domain = int(current_variants.get("in_domain", pd.Series()).fillna(0).sum())
        mean_af = current_variants.get("gnomad_max_af", pd.Series()).dropna().mean()

        summary_text = f"""
**Summary Statistics**

- **Filtered variants:** {total:,}
- **Domain-overlapping variants:** {inside_domain:,}
- **Mean gnomAD AF:** {mean_af:.5f if not np.isnan(mean_af) else 'N/A'}
"""
        summary_output = mo.md(summary_text)
    
    return summary_output,

@app.cell
def __(summary_output):
    summary_output
    return

@app.cell
def __(current_variants, mo, top_k_slider):
    if current_variants.empty:
        variant_table_output = mo.md("No variants met the criteria.")
    else:
        display_cols = [
            "variant_id",
            "clinical_significance",
            "coding_impact",
            "regulatory_region",
            "gnomad_max_af",
            "spliceai_max_score",
            "missense_combined_score",
            "conservation_score",
            "composite_score",
        ]
        available = [c for c in display_cols if c in current_variants.columns]
        preview = current_variants[available].head(top_k_slider.value)
        variant_table_output = mo.ui.dataframe(preview)
    return variant_table_output,

@app.cell
def __(variant_table_output):
    variant_table_output
    return

@app.cell
def __(current_variants, mo):
    if current_variants.empty:
        distribution_output = mo.md("Nothing to visualize yet – adjust your filters.")
    else:
        try:
            import plotly.express as px

            plots = {}

            if "gnomad_max_af" in current_variants.columns:
                plots["Allele frequency"] = px.histogram(
                    current_variants,
                    x="gnomad_max_af",
                    nbins=40,
                    title="gnomAD AF",
                )

            if "clinical_significance" in current_variants.columns:
                plots["Clinical significance"] = px.pie(
                    current_variants,
                    names="clinical_significance",
                    title="Clinical significance mix",
                )

            if "regulatory_type" in current_variants.columns:
                plots["Regulatory region"] = px.bar(
                    current_variants["regulatory_type"].value_counts().reset_index(),
                    x="index",
                    y="regulatory_type",
                    labels={"index": "Type", "regulatory_type": "Count"},
                    title="Regulatory context",
                )

            distribution_output = mo.ui.tabs(plots) if plots else mo.md("Add gnomAD/regulatory columns to plot.")
        except ImportError:
            distribution_output = mo.md("Install plotly to visualize distributions.")
    
    return distribution_output,

@app.cell
def __(distribution_output):
    distribution_output
    return

@app.cell
def __(current_variants, mo):
    if current_variants.empty or "domain_label" not in current_variants.columns:
        domain_output = mo.md("Domain annotations unavailable.")
    else:
        counts = current_variants["domain_label"].fillna("Outside domain").value_counts().reset_index()
        counts.columns = ["Domain", "Variants"]
        domain_output = mo.ui.table(counts)
    
    return domain_output,

@app.cell
def __(domain_output):
    domain_output
    return

@app.cell
def __(current_variants, mo, top_k_slider):
    if current_variants.empty:
        downloads_output = mo.md("Nothing to export.")
    else:
        top = current_variants.head(top_k_slider.value)
        downloads_output = mo.hstack(
            [
                mo.download(top, label="Download CSV", filename="abca4_top_variants.csv"),
                mo.download(top, label="Download Parquet", filename="abca4_top_variants.parquet"),
            ]
        )
    return downloads_output,

@app.cell
def __(downloads_output):
    downloads_output
    return

@app.cell
def __(mo):
    mo.md(
        """
### Tips
- Run `invoke features.compute` whenever new raw data lands, then refresh this notebook.
- Use the ranking weights to mimic optimization objectives before invoking Strand strategies.
- Export the top variants directly as CSV/Parquet for ML or reporting notebooks.
"""
    )
    return

if __name__ == "__main__":
    app.run()
