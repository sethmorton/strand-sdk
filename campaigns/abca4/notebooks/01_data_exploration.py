#!/usr/bin/env python3
"""ABCA4 Campaign – interactive data exploration with Marimo."""

import marimo as mo
import numpy as np
import pandas as pd
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
FEATURE_MATRIX = CAMPAIGN_ROOT / "data_processed" / "features" / "abca4_feature_matrix.parquet"

mo.md("# 🔬 ABCA4 Variant Explorer")
mo.md(
    """
Use the controls below to slice the unified ABCA4 feature matrix. Every widget is
reactive: once the data is filtered, tables, charts, and download buttons update
automatically.
"""
)


@mo.cell
def feature_matrix():
    if not FEATURE_MATRIX.exists():
        print(f"⚠️ Missing feature matrix at {FEATURE_MATRIX}. Run `invoke features.compute`. ")
        return pd.DataFrame()

    df = pd.read_parquet(FEATURE_MATRIX)
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    return df


df_features = feature_matrix()


def _safe_unique(series: pd.Series) -> list[str]:
    if series.empty:
        return []
    return sorted(series.dropna().astype(str).unique())


@mo.cell
def filter_controls(df_features: pd.DataFrame):
    if df_features.empty:
        placeholder = mo.card("Filters", mo.md("Upload data to unlock controls."))
        return placeholder, None, None, None, None, None, None, None

    clinical = mo.ui.multiselect(
        options=_safe_unique(df_features["clinical_significance"]),
        value=_safe_unique(df_features["clinical_significance"]),
        label="Clinical Significance",
    )

    coding = mo.ui.multiselect(
        options=_safe_unique(df_features["coding_impact"]),
        value=_safe_unique(df_features["coding_impact"]),
        label="Coding Impact",
    )

    regulatory = mo.ui.multiselect(
        options=_safe_unique(df_features["regulatory_type"]),
        value=_safe_unique(df_features["regulatory_type"]),
        label="Regulatory Context",
    )

    domain_only = mo.ui.switch(False, label="Only keep variants inside annotated domains")

    max_af = float(df_features.get("gnomad_max_af", pd.Series([0.0])).fillna(0).max() or 0.01)
    af_slider = mo.ui.slider(0.0, max(0.01, max_af), value=min(0.01, max_af), step=0.0001, label="Max gnomAD AF")

    splice_slider = mo.ui.slider(0.0, 1.0, value=0.1, step=0.01, label="Min SpliceAI score")
    missense_slider = mo.ui.slider(0.0, 1.0, value=0.3, step=0.01, label="Min missense score")

    panel = mo.card(
        "Filters",
        mo.vstack(
            [
                mo.hstack([clinical, coding]),
                mo.hstack([regulatory, domain_only]),
                mo.hstack([af_slider, splice_slider, missense_slider]),
            ]
        ),
    )

    return panel, clinical, coding, regulatory, domain_only, af_slider, splice_slider, missense_slider


(
    filter_panel,
    clinical_significance,
    coding_impacts,
    regulatory_types,
    domain_only,
    allele_frequency_max,
    splice_threshold,
    missense_threshold,
) = filter_controls(df_features)
filter_panel


@mo.cell
def weight_controls():
    reg = mo.ui.slider(0.0, 1.0, value=0.35, step=0.05, label="Regulatory weight")
    splice = mo.ui.slider(0.0, 1.0, value=0.25, step=0.05, label="Splice weight")
    missense = mo.ui.slider(0.0, 1.0, value=0.25, step=0.05, label="Missense weight")
    conservation = mo.ui.slider(0.0, 1.0, value=0.15, step=0.05, label="Conservation weight")

    panel = mo.card(
        "Ranking weights",
        mo.vstack([
            mo.md("Tune the scoring weights (will normalize automatically)."),
            mo.hstack([reg, splice, missense, conservation]),
        ]),
    )

    return panel, reg, splice, missense, conservation


weights_panel, reg_weight, splice_weight, missense_weight, conservation_weight = weight_controls()
weights_panel


top_k_slider = mo.ui.slider(5, 200, value=50, step=5, label="Top-K for downloads")
top_k_slider


@mo.cell
def normalized_weights(reg_weight, splice_weight, missense_weight, conservation_weight):
    sliders = [reg_weight, splice_weight, missense_weight, conservation_weight]
    values = np.array([slider.value if slider else 0.25 for slider in sliders], dtype=float)
    if values.sum() == 0:
        values = np.ones_like(values)
    values = values / values.sum()
    labels = [
        ("regulatory_score", values[0]),
        ("spliceai_max_score", values[1]),
        ("missense_combined_score", values[2]),
        ("conservation_score", values[3]),
    ]
    return dict(labels)


score_weights = normalized_weights(reg_weight, splice_weight, missense_weight, conservation_weight)


@mo.cell
def filtered_variants(
    df_features,
    clinical_significance,
    coding_impacts,
    regulatory_types,
    domain_only,
    allele_frequency_max,
    splice_threshold,
    missense_threshold,
    score_weights,
):
    df = df_features.copy()
    if df.empty:
        return df

    if clinical_significance:
        selected = clinical_significance.value or clinical_significance.options
        df = df[df["clinical_significance"].isin(selected)]

    if coding_impacts:
        selected = coding_impacts.value or coding_impacts.options
        df = df[df["coding_impact"].isin(selected)]

    if regulatory_types:
        selected = regulatory_types.value or regulatory_types.options
        df = df[df["regulatory_type"].isin(selected)]

    if domain_only and domain_only.value:
        df = df[df.get("in_domain", 0) == 1]

    if allele_frequency_max:
        df = df[df.get("gnomad_max_af", 0).fillna(0) <= allele_frequency_max.value]

    if splice_threshold:
        df = df[df.get("spliceai_max_score", 0).fillna(0) >= splice_threshold.value]

    if missense_threshold:
        df = df[df.get("missense_combined_score", 0).fillna(0) >= missense_threshold.value]

    for feature, weight in score_weights.items():
        df[feature] = df.get(feature, 0).fillna(0)

    df = df.assign(
        composite_score=sum(weight * df[feature] for feature, weight in score_weights.items())
    )

    df = df.sort_values("composite_score", ascending=False)
    return df.reset_index(drop=True)


current_variants = filtered_variants(
    df_features,
    clinical_significance,
    coding_impacts,
    regulatory_types,
    domain_only,
    allele_frequency_max,
    splice_threshold,
    missense_threshold,
    score_weights,
)


@mo.cell
def summary_cards(df_features, current_variants):
    if df_features.empty:
        return mo.card("Summary", mo.md("No data loaded."))

    total = len(current_variants)
    inside_domain = int(current_variants.get("in_domain", pd.Series()).fillna(0).sum())
    mean_af = current_variants.get("gnomad_max_af", pd.Series()).dropna().mean()

    content = mo.vstack(
        [
            mo.md(f"**Filtered variants:** {total:,}"),
            mo.md(f"**Domain-overlapping variants:** {inside_domain:,}"),
            mo.md(f"**Mean gnomAD AF:** {mean_af:.5f}" if not np.isnan(mean_af) else "**Mean gnomAD AF:** N/A"),
        ]
    )

    return mo.card("Snapshot", content)


summary_cards(df_features, current_variants)


@mo.cell
def variant_table(current_variants, top_k_slider):
    if current_variants.empty:
        return mo.md("No variants met the criteria.")

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
    return mo.ui.dataframe(preview)


variant_table(current_variants, top_k_slider)


@mo.cell
def distribution_charts(current_variants):
    if current_variants.empty:
        return mo.md("Nothing to visualize yet – adjust your filters.")

    import plotly.express as px

    plots = {}

    if "gnomad_max_af" in current_variants:
        plots["Allele frequency"] = px.histogram(
            current_variants,
            x="gnomad_max_af",
            nbins=40,
            title="gnomAD AF",
        )

    if "clinical_significance" in current_variants:
        plots["Clinical significance"] = px.pie(
            current_variants,
            names="clinical_significance",
            title="Clinical significance mix",
        )

    if "regulatory_type" in current_variants:
        plots["Regulatory region"] = px.bar(
            current_variants["regulatory_type"].value_counts().reset_index(),
            x="index",
            y="regulatory_type",
            labels={"index": "Type", "regulatory_type": "Count"},
            title="Regulatory context",
        )

    return mo.ui.tabs(plots) if plots else mo.md("Add gnomAD/regulatory columns to plot.")


distribution_charts(current_variants)


@mo.cell
def domain_breakdown(current_variants):
    if current_variants.empty or "domain_label" not in current_variants:
        return mo.md("Domain annotations unavailable.")

    counts = current_variants["domain_label"].fillna("Outside domain").value_counts().reset_index()
    counts.columns = ["Domain", "Variants"]
    return mo.card("Domain breakdown", mo.ui.table(counts))


domain_breakdown(current_variants)


@mo.cell
def download_top_variants(current_variants, top_k_slider):
    if current_variants.empty:
        return mo.md("Nothing to export.")

    top = current_variants.head(top_k_slider.value)
    return mo.hstack(
        [
            mo.download(top, label="Download CSV", filename="abca4_top_variants.csv"),
            mo.download(top, label="Download Parquet", filename="abca4_top_variants.parquet"),
        ]
    )


download_top_variants(current_variants, top_k_slider)


mo.md(
    """
### Tips
- Run `invoke features.compute` whenever new raw data lands, then refresh this notebook.
- Use the ranking weights to mimic optimization objectives before invoking Strand strategies.
- Export the top variants directly as CSV/Parquet for ML or reporting notebooks.
"""
)
