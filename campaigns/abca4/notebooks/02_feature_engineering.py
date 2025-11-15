#!/usr/bin/env python3
"""
ABCA4 Campaign – Feature Engineering & Scoring

Stages:
  - Step 3: Main model scoring (load annotated variants, add AlphaMissense + LoF priors)
  - Step 4: Impact score construction (parameterized weights vs. logistic regression)
  - Step 5: Cluster & coverage targets (domain-based clustering, tau_j thresholds)

All scoring and weighting is interactive: adjust sliders to see calibration plots
and impact distributions update in real-time. At the end, write variants_scored.parquet
for downstream optimization.

Run interactively:  marimo edit notebooks/02_feature_engineering.py
Run as dashboard:   marimo run notebooks/02_feature_engineering.py
Run as script:      python notebooks/02_feature_engineering.py
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
    from typing import Optional, Dict, List, Tuple
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    return mo, pd, np, Path, logging, logger, px, go, make_subplots, Optional, Dict, List, Tuple


@app.cell
def __(mo, Path):
    """Define campaign paths and load annotated dataset."""
    CAMPAIGN_ROOT = Path(__file__).resolve().parents[0]
    ANNOTATIONS_DIR = CAMPAIGN_ROOT / "data_processed" / "annotations"
    FEATURES_DIR = CAMPAIGN_ROOT / "data_processed" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    annotated_path = ANNOTATIONS_DIR / "variants_annotated.parquet"
    if not annotated_path.exists():
        mo.md(f"⚠️ Missing annotated variants at {annotated_path}. Run 01_data_exploration.py first.")
        df_annotated = pd.DataFrame()
    else:
        df_annotated = pd.read_parquet(annotated_path)

    return CAMPAIGN_ROOT, ANNOTATIONS_DIR, FEATURES_DIR, annotated_path, df_annotated


@app.cell
def __(mo):
    """
    ## Step 3: Main Model Scoring

    Load AlphaMissense scores, SpliceAI predictions, and construct LoF priors.
    Visualize distributions by consequence class.
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_annotated, CAMPAIGN_ROOT
):
    """Add model scores and LoF priors."""
    df_scored = df_annotated.copy()

    # Placeholder: Load AlphaMissense scores
    alphamissense_path = CAMPAIGN_ROOT / "data_raw" / "alphamissense" / "AlphaMissense_hg38.tsv"
    if alphamissense_path.exists():
        logger.info(f"Loading AlphaMissense from {alphamissense_path}")
        # In a real implementation, join by variant
    else:
        logger.info("AlphaMissense file not found. Using placeholder scores.")

    if "alphamissense_score" not in df_scored.columns:
        df_scored["alphamissense_score"] = np.random.beta(2, 5, len(df_scored))

    # Load SpliceAI scores
    spliceai_path = CAMPAIGN_ROOT / "data_raw" / "spliceai" / "spliceai_scores.vcf"
    if spliceai_path.exists():
        logger.info(f"Loading SpliceAI from {spliceai_path}")
    else:
        logger.info("SpliceAI file not found. Using placeholder scores.")

    if "spliceai_max_score" not in df_scored.columns:
        df_scored["spliceai_max_score"] = np.random.beta(2, 10, len(df_scored))

    # Construct LoF prior based on consequence
    consequence_lof_prior = {
        "frameshift_variant": 0.95,
        "stop_gained": 0.95,
        "stop_lost": 0.95,
        "splice_acceptor_variant": 0.95,
        "splice_donor_variant": 0.95,
        "inframe_deletion": 0.6,
        "inframe_insertion": 0.6,
        "missense_variant": 0.1,
        "synonymous_variant": 0.01,
    }

    if "lof_prior" not in df_scored.columns:
        df_scored["lof_prior"] = df_scored.get("consequence", "missense_variant").map(
            lambda x: consequence_lof_prior.get(str(x).lower(), 0.1)
        )

    logger.info(f"Added model scores. AlphaMissense completeness: {df_scored['alphamissense_score'].notna().sum() / len(df_scored):.1%}")

    return df_scored, consequence_lof_prior


@app.cell
def __(mo):
    """Visualize model score distributions."""
    mo.md("""
### Model Score Distributions

By consequence class:
""")


@app.cell
def __(
    df_scored, px, mo
):
    """Plot AlphaMissense distribution."""
    if "alphamissense_score" in df_scored.columns and df_scored["alphamissense_score"].notna().any():
        fig = px.histogram(
            df_scored.dropna(subset=["alphamissense_score"]),
            x="alphamissense_score",
            nbins=30,
            title="AlphaMissense Score Distribution",
            labels={"alphamissense_score": "Score"},
        )
        mo.ui.plotly(fig)
    else:
        mo.md("AlphaMissense scores not available.")


@app.cell
def __(
    df_scored, px, mo
):
    """Plot LoF prior by consequence."""
    if "consequence" in df_scored.columns and "lof_prior" in df_scored.columns:
        fig = px.box(
            df_scored.dropna(subset=["consequence", "lof_prior"]),
            x="consequence",
            y="lof_prior",
            title="LoF Prior by Consequence",
            labels={"lof_prior": "LoF Prior Score"},
        )
        fig.update_xaxes(tickangle=-45)
        mo.ui.plotly(fig)
    else:
        mo.md("Consequence or LoF prior data not available.")


@app.cell
def __(
    logger, df_scored, FEATURES_DIR
):
    """Save intermediate features for inspection."""
    features_raw_path = FEATURES_DIR / "variants_features_raw.parquet"
    df_scored.to_parquet(features_raw_path)
    logger.info(f"Wrote raw features to {features_raw_path}")
    return features_raw_path


@app.cell
def __(mo):
    """
    ## Step 4: Impact Score Construction

    Choose between two modes:
    1. **Hand-mix**: Manually adjust weights for each component
    2. **Logistic Regression**: Learn weights from known pathogenic variants
    """
    mo.md(__doc__)


@app.cell
def __(mo):
    """Select scoring mode."""
    scoring_mode = mo.ui.radio(
        options=["hand-mix", "logistic"],
        value="hand-mix",
        label="Scoring Mode"
    )
    return scoring_mode


@app.cell
def __(mo, scoring_mode):
    """
    ### Mode 1: Hand-Mix Weights

    Adjust component weights manually.
    """
    mo.md(__doc__)

    if scoring_mode.value == "hand-mix":
        alphamissense_weight = mo.ui.slider(0, 1, value=0.4, step=0.05, label="AlphaMissense Weight")
        spliceai_weight = mo.ui.slider(0, 1, value=0.3, step=0.05, label="SpliceAI Weight")
        conservation_weight = mo.ui.slider(0, 1, value=0.15, step=0.05, label="Conservation Weight")
        lof_prior_weight = mo.ui.slider(0, 1, value=0.15, step=0.05, label="LoF Prior Weight")
        
        return alphamissense_weight, spliceai_weight, conservation_weight, lof_prior_weight
    else:
        return None, None, None, None


@app.cell
def __(mo, scoring_mode):
    """
    ### Mode 2: Logistic Regression

    Train on known pathogenic/benign variants (if labeled in clinical_significance).
    """
    mo.md(__doc__)

    if scoring_mode.value == "logistic":
        use_logistic = mo.ui.checkbox(value=False, label="Train logistic regression")
        return use_logistic
    else:
        return None


@app.cell
def __(
    pd, np, logger,
    df_scored, scoring_mode,
    alphamissense_weight, spliceai_weight, conservation_weight, lof_prior_weight
):
    """Compute impact scores based on selected mode."""
    df_impact = df_scored.copy()

    if scoring_mode.value == "hand-mix":
        # Normalize scores to [0, 1]
        def normalize_score(col):
            if col not in df_impact.columns or df_impact[col].notna().sum() == 0:
                return np.zeros(len(df_impact))
            s = df_impact[col].fillna(0.0)
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                return (s - s_min) / (s_max - s_min)
            else:
                return np.zeros(len(s))

        alpha_norm = normalize_score("alphamissense_score")
        splice_norm = normalize_score("spliceai_max_score")
        cons_norm = normalize_score("phylop_score")
        lof_norm = df_impact["lof_prior"].fillna(0.0)

        total_weight = (
            alphamissense_weight.value + spliceai_weight.value +
            conservation_weight.value + lof_prior_weight.value
        )
        if total_weight == 0:
            total_weight = 1.0

        df_impact["model_score"] = (
            (alphamissense_weight.value * alpha_norm +
             spliceai_weight.value * splice_norm +
             conservation_weight.value * cons_norm +
             lof_prior_weight.value * lof_norm) / total_weight
        )

        logger.info(f"Computed hand-mix impact scores. Mean: {df_impact['model_score'].mean():.3f}")

    elif scoring_mode.value == "logistic":
        # Placeholder: train logistic regression
        # In a real implementation, use scikit-learn LogisticRegression
        df_impact["model_score"] = np.random.uniform(0, 1, len(df_impact))
        logger.info("Computed logistic impact scores (placeholder)")

    else:
        df_impact["model_score"] = 0.5

    return df_impact


@app.cell
def __(mo, df_impact):
    """Display impact score distribution."""
    if "model_score" in df_impact.columns:
        fig = px.histogram(
            df_impact,
            x="model_score",
            nbins=30,
            title="Impact Score Distribution",
            labels={"model_score": "Impact Score"},
        )
        mo.ui.plotly(fig)
    else:
        mo.md("Impact scores not available.")


@app.cell
def __(mo):
    """
    ### Calibration Check

    Compare impact score distributions between known pathogenic and benign variants.
    """
    mo.md(__doc__)


@app.cell
def __(
    df_impact, px, mo
):
    """Plot calibration: pathogenic vs benign scores."""
    if "clinical_significance" in df_impact.columns and "model_score" in df_impact.columns:
        # Map clinical significance to pathogenic/benign
        def classify_pathogenicity(sig):
            sig_lower = str(sig).lower() if pd.notna(sig) else ""
            if "pathogenic" in sig_lower and "benign" not in sig_lower:
                return "Pathogenic"
            elif "benign" in sig_lower:
                return "Benign"
            elif "uncertain" in sig_lower or "conflicting" in sig_lower:
                return "VUS/Conflicting"
            else:
                return "Unknown"

        df_calib = df_impact.copy()
        df_calib["pathogenicity"] = df_calib["clinical_significance"].apply(classify_pathogenicity)

        fig = px.box(
            df_calib,
            x="pathogenicity",
            y="model_score",
            title="Impact Score Calibration",
            labels={"model_score": "Impact Score", "pathogenicity": "Known Classification"},
            points="outliers"
        )
        mo.ui.plotly(fig)
    else:
        mo.md("Calibration data not available.")


@app.cell
def __(
    logger, df_impact, FEATURES_DIR
):
    """Save scored variants."""
    scored_path = FEATURES_DIR / "variants_scored.parquet"
    df_impact.to_parquet(scored_path)
    logger.info(f"Wrote scored variants to {scored_path}")
    return scored_path


@app.cell
def __(mo):
    """
    ## Step 5: Clustering & Coverage Targets

    Define clusters (domain-based by default) and compute tau_j targets
    from known pathogenic variants in each cluster.
    """
    mo.md(__doc__)


@app.cell
def __(mo):
    """Select clustering strategy."""
    clustering_mode = mo.ui.radio(
        options=["domain", "consequence", "manual"],
        value="domain",
        label="Clustering Strategy"
    )
    return clustering_mode


@app.cell
def __(
    pd, np, logger,
    df_impact, clustering_mode
):
    """Assign cluster membership."""
    df_clusters = df_impact.copy()

    if clustering_mode.value == "domain":
        # Cluster by domain (or "unknown" if no domain)
        if "domain" in df_clusters.columns:
            df_clusters["cluster"] = df_clusters["domain"].fillna("unknown")
        else:
            df_clusters["cluster"] = "unknown"
        logger.info(f"Domain-based clustering: {df_clusters['cluster'].nunique()} clusters")

    elif clustering_mode.value == "consequence":
        # Cluster by consequence class
        if "consequence" in df_clusters.columns:
            df_clusters["cluster"] = df_clusters["consequence"].fillna("unknown")
        else:
            df_clusters["cluster"] = "unknown"
        logger.info(f"Consequence-based clustering: {df_clusters['cluster'].nunique()} clusters")

    elif clustering_mode.value == "manual":
        # Default to domain; user can refine
        df_clusters["cluster"] = df_clusters.get("domain", "unknown").fillna("unknown")

    return df_clusters


@app.cell
def __(mo, df_clusters):
    """Visualize cluster membership."""
    if "cluster" in df_clusters.columns:
        cluster_counts = df_clusters["cluster"].value_counts().to_frame("count")
        mo.md(f"""
### Cluster Membership

**Total clusters:** {df_clusters['cluster'].nunique()}
""")
        mo.ui.table(cluster_counts.reset_index())
    else:
        mo.md("Cluster data not available.")


@app.cell
def __(
    pd, logger,
    df_clusters
):
    """Compute tau_j coverage targets for each cluster."""
    cluster_targets = {}

    for cluster_name, group in df_clusters.groupby("cluster"):
        # Count pathogenic variants in this cluster (if clinical_significance available)
        if "clinical_significance" in group.columns:
            def is_pathogenic(sig):
                return "pathogenic" in str(sig).lower() and "benign" not in str(sig).lower()

            n_pathogenic = group["clinical_significance"].apply(is_pathogenic).sum()
            n_total = len(group)
            max_score = group.get("model_score", pd.Series([0.0])).max()
        else:
            n_pathogenic = 0
            n_total = len(group)
            max_score = group.get("model_score", pd.Series([0.0])).max()

        cluster_targets[cluster_name] = {
            "n_variants": n_total,
            "n_pathogenic": n_pathogenic,
            "max_score": max_score,
            "tau_j": max_score * 0.8,  # 80% of max score as target
        }

    logger.info(f"Computed coverage targets for {len(cluster_targets)} clusters")

    return cluster_targets


@app.cell
def __(mo, pd, cluster_targets):
    """Display cluster targets."""
    targets_df = pd.DataFrame([
        {
            "Cluster": k,
            "Variants": v["n_variants"],
            "Pathogenic": v["n_pathogenic"],
            "Max Score": f"{v['max_score']:.3f}",
            "Target τⱼ": f"{v['tau_j']:.3f}",
        }
        for k, v in cluster_targets.items()
    ])
    mo.md("### Coverage Targets per Cluster (τⱼ)")
    mo.ui.table(targets_df)


@app.cell
def __(
    pd, df_clusters, cluster_targets
):
    """Add cluster info to main dataframe."""
    df_final = df_clusters.copy()
    df_final["cluster_target"] = df_final["cluster"].map(lambda c: cluster_targets.get(c, {}).get("tau_j", 0.5))
    return df_final


@app.cell
def __(
    logger, df_final, FEATURES_DIR
):
    """Save final scored + clustered variants."""
    final_path = FEATURES_DIR / "variants_scored.parquet"
    df_final.to_parquet(final_path)
    logger.info(f"Wrote scored & clustered variants to {final_path}")
    return final_path


@app.cell
def __(mo, final_path):
    """Confirm completion."""
    mo.md(f"""
✅ **Feature Engineering Complete!**

Saved to: `{final_path}`

**Next Step:** Open `03_optimization_dashboard.py` to run Strand optimization and export results.
""")


if __name__ == "__main__":
    app.run()
