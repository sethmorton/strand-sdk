#!/usr/bin/env python3
"""
ABCA4 Campaign – Feature Engineering & Scoring

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

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    return mo, pd, np, Path, logging, logger, Optional, Dict, List, Tuple


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
    """
    mo.md(__doc__)


@app.cell
def __(
    pd, np, logger,
    df_annotated, CAMPAIGN_ROOT
):
    """Add model scores and LoF priors."""
    df_scored_step3 = df_annotated.copy()

    # Add placeholder scores for testing
    if "alphamissense_score" not in df_scored_step3.columns:
        df_scored_step3["alphamissense_score"] = np.random.beta(2, 5, len(df_scored_step3))

    if "spliceai_max_score" not in df_scored_step3.columns:
        df_scored_step3["spliceai_max_score"] = np.random.beta(2, 10, len(df_scored_step3))

    # Construct LoF prior based on consequence
    _consequence_lof = {
        "frameshift_variant": 0.95,
        "stop_gained": 0.95,
        "splice_acceptor_variant": 0.95,
        "splice_donor_variant": 0.95,
        "missense_variant": 0.1,
        "synonymous_variant": 0.01,
    }

    if "lof_prior" not in df_scored_step3.columns:
        df_scored_step3["lof_prior"] = df_scored_step3.get("consequence", "missense_variant").map(
            lambda x: _consequence_lof.get(str(x).lower(), 0.1)
        )

    logger.info(f"Added model scores.")

    return df_scored_step3


@app.cell
def __(
    pd, df_scored_step3, logger, FEATURES_DIR
):
    """Save intermediate features."""
    _features_raw_path = FEATURES_DIR / "variants_features_raw.parquet"
    df_scored_step3.to_parquet(_features_raw_path)
    logger.info(f"Wrote raw features to {_features_raw_path}")
    return _features_raw_path


@app.cell
def __(mo):
    """
    ## Step 4: Impact Score Construction

    Choose between two modes: hand-mix or logistic regression.
    """
    mo.md(__doc__)

    scoring_mode_widget = mo.ui.radio(
        options=["hand-mix", "logistic"],
        value="hand-mix",
        label="Scoring Mode"
    )

    return scoring_mode_widget


@app.cell
def __(mo, scoring_mode_widget):
    """Create hand-mix weight sliders."""
    if scoring_mode_widget.value == "hand-mix":
        alpha_wgt = mo.ui.slider(0, 1, value=0.4, step=0.05, label="AlphaMissense Weight")
        splice_wgt = mo.ui.slider(0, 1, value=0.3, step=0.05, label="SpliceAI Weight")
        cons_wgt = mo.ui.slider(0, 1, value=0.15, step=0.05, label="Conservation Weight")
        lof_wgt = mo.ui.slider(0, 1, value=0.15, step=0.05, label="LoF Prior Weight")
    else:
        alpha_wgt = None
        splice_wgt = None
        cons_wgt = None
        lof_wgt = None

    return alpha_wgt, splice_wgt, cons_wgt, lof_wgt


@app.cell
def __(
    pd, np, logger,
    df_scored_step3, scoring_mode_widget,
    alpha_wgt, splice_wgt, cons_wgt, lof_wgt
):
    """Compute impact scores."""
    df_impact = df_scored_step3.copy()

    if scoring_mode_widget.value == "hand-mix" and alpha_wgt is not None:
        # Normalize scores to [0, 1]
        def _normalize_score(col):
            if col not in df_impact.columns or df_impact[col].notna().sum() == 0:
                return np.zeros(len(df_impact))
            s = df_impact[col].fillna(0.0)
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                return (s - s_min) / (s_max - s_min)
            else:
                return np.zeros(len(s))

        _alpha_norm = _normalize_score("alphamissense_score")
        _splice_norm = _normalize_score("spliceai_max_score")
        _cons_norm = _normalize_score("phylop_score")
        _lof_norm = df_impact["lof_prior"].fillna(0.0)

        _total_wgt = (
            alpha_wgt.value + splice_wgt.value +
            cons_wgt.value + lof_wgt.value
        )
        if _total_wgt == 0:
            _total_wgt = 1.0

        df_impact["model_score"] = (
            (alpha_wgt.value * _alpha_norm +
             splice_wgt.value * _splice_norm +
             cons_wgt.value * _cons_norm +
             lof_wgt.value * _lof_norm) / _total_wgt
        )

        logger.info(f"Computed hand-mix impact scores.")

    else:
        # Placeholder logistic mode
        df_impact["model_score"] = np.random.uniform(0, 1, len(df_impact))
        logger.info("Computed logistic impact scores (placeholder)")

    return df_impact


@app.cell
def __(mo, go, df_impact):
    """Visualize impact score distribution with plotly."""
    if "model_score" in df_impact.columns and not df_impact.empty:
        try:
            _fig_score = go.Figure()
            _fig_score.add_trace(go.Histogram(
                x=df_impact["model_score"].dropna(),
                nbinsx=30,
                name="Impact Score",
                marker_color="rgba(99, 110, 250, 0.7)"
            ))
            _fig_score.update_layout(
                title="Impact Score Distribution",
                xaxis_title="Impact Score",
                yaxis_title="Frequency",
                hovermode="x unified",
                showlegend=False,
                template="plotly_white"
            )
            mo.ui.plotly(_fig_score)
        except Exception as _e:
            mo.md(f"Visualization error: {_e}")
    else:
        mo.md("Impact scores not available.")


@app.cell
def __(mo):
    """
    ## Step 5: Clustering & Coverage Targets

    Define clusters and compute coverage thresholds.
    """
    mo.md(__doc__)

    clustering_widget = mo.ui.radio(
        options=["domain", "consequence", "manual"],
        value="domain",
        label="Clustering Strategy"
    )

    return clustering_widget


@app.cell
def __(
    pd, np, logger,
    df_impact, clustering_widget
):
    """Assign cluster membership."""
    df_clusters = df_impact.copy()

    if clustering_widget.value == "domain":
        if "domain" in df_clusters.columns:
            df_clusters["cluster"] = df_clusters["domain"].fillna("unknown")
        else:
            df_clusters["cluster"] = "unknown"
        logger.info(f"Domain-based clustering: {df_clusters['cluster'].nunique()} clusters")

    elif clustering_widget.value == "consequence":
        if "consequence" in df_clusters.columns:
            df_clusters["cluster"] = df_clusters["consequence"].fillna("unknown")
        else:
            df_clusters["cluster"] = "unknown"
        logger.info(f"Consequence-based clustering: {df_clusters['cluster'].nunique()} clusters")

    else:
        df_clusters["cluster"] = df_clusters.get("domain", "unknown").fillna("unknown")

    return df_clusters


@app.cell
def __(mo, df_clusters):
    """Display cluster membership."""
    if "cluster" in df_clusters.columns and not df_clusters.empty:
        _cluster_counts = df_clusters["cluster"].value_counts().to_frame("count")
        mo.md(f"""
### Cluster Membership

**Total clusters:** {df_clusters['cluster'].nunique()}
""")
        mo.ui.table(_cluster_counts.reset_index())


@app.cell
def __(
    pd, logger,
    df_clusters
):
    """Compute cluster coverage targets."""
    _cluster_targets = {}

    for _cluster_name, _group in df_clusters.groupby("cluster"):
        # Count pathogenic variants
        if "clinical_significance" in _group.columns:
            def _is_pathogenic(sig):
                return "pathogenic" in str(sig).lower() and "benign" not in str(sig).lower()
            _n_pathogenic = _group["clinical_significance"].apply(_is_pathogenic).sum()
        else:
            _n_pathogenic = 0

        _n_total = len(_group)
        _max_score = _group.get("model_score", pd.Series([0.0])).max()

        _cluster_targets[_cluster_name] = {
            "n_variants": _n_total,
            "n_pathogenic": _n_pathogenic,
            "max_score": _max_score,
            "tau_j": _max_score * 0.8,
        }

    logger.info(f"Computed coverage targets for {len(_cluster_targets)} clusters")

    return _cluster_targets


@app.cell
def __(mo, pd, df_clusters):
    """Display cluster targets."""
    _cluster_targets_display = {}

    for _cluster_name, _group in df_clusters.groupby("cluster"):
        if "clinical_significance" in _group.columns:
            def _is_path(sig):
                return "pathogenic" in str(sig).lower() and "benign" not in str(sig).lower()
            _n_path = _group["clinical_significance"].apply(_is_path).sum()
        else:
            _n_path = 0

        _n_tot = len(_group)
        _max_sc = _group.get("model_score", pd.Series([0.0])).max()

        _cluster_targets_display[_cluster_name] = {
            "n_variants": _n_tot,
            "n_pathogenic": _n_path,
            "max_score": _max_sc,
            "tau_j": _max_sc * 0.8,
        }

    _targets_df = pd.DataFrame([
        {
            "Cluster": k,
            "Variants": v["n_variants"],
            "Pathogenic": v["n_pathogenic"],
            "Max Score": f"{v['max_score']:.3f}",
            "Target τⱼ": f"{v['tau_j']:.3f}",
        }
        for k, v in _cluster_targets_display.items()
    ])
    mo.md("### Coverage Targets per Cluster")
    mo.ui.table(_targets_df)


@app.cell
def __(
    pd, df_clusters
):
    """Add cluster info and finalize."""
    df_final_scored = df_clusters.copy()

    # Compute cluster targets inline
    _cluster_tgt_dict = {}
    for _cn, _cg in df_clusters.groupby("cluster"):
        _mx = _cg.get("model_score", pd.Series([0.0])).max()
        _cluster_tgt_dict[_cn] = _mx * 0.8

    df_final_scored["cluster_target"] = df_final_scored["cluster"].map(
        lambda c: _cluster_tgt_dict.get(c, 0.5)
    )
    return df_final_scored


@app.cell
def __(
    logger, df_final_scored, FEATURES_DIR
):
    """Save final scored and clustered variants."""
    _final_path = FEATURES_DIR / "variants_scored.parquet"
    df_final_scored.to_parquet(_final_path)
    logger.info(f"Wrote scored & clustered variants to {_final_path}")
    return _final_path


@app.cell
def __(mo, logger, df_final_scored, FEATURES_DIR):
    """Confirm completion and save."""
    _final_path_confirm = FEATURES_DIR / "variants_scored.parquet"
    df_final_scored.to_parquet(_final_path_confirm)
    logger.info(f"Wrote scored & clustered variants")
    
    mo.md(f"""
✅ **Feature Engineering Complete!**

Saved to: `{_final_path_confirm}`

**Next Step:** Open `03_optimization_dashboard.py` for Strand optimization.
""")


if __name__ == "__main__":
    app.run()
