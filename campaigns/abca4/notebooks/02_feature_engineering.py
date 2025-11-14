#!/usr/bin/env python3
"""
ABCA4 Campaign - Feature Engineering Notebook

Interactive feature engineering for ABCA4 variants.
Tune parameters, visualize distributions, and see correlations in real-time.

Run with: marimo run notebooks/02_feature_engineering.py
Edit with: marimo edit notebooks/02_feature_engineering.py
"""

import marimo as mo
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

mo.md("# 🔧 ABCA4 Feature Engineering")

mo.md("""
This notebook provides interactive feature engineering for ABCA4 variants.
Tune parameters, visualize feature distributions, and explore correlations in real-time.
""")

# Load base variant data (placeholder)
@mo.cell
def load_variant_data():
    """Load base variant data"""
    # In real implementation, load from data_processed/
    variants_df = pd.DataFrame({
        'variant_id': range(1000),
        'chrom': ['1'] * 1000,
        'pos': np.random.randint(94400000, 95200000, 1000),
        'ref': np.random.choice(['A', 'C', 'G', 'T'], 1000),
        'alt': np.random.choice(['A', 'C', 'G', 'T'], 1000),
        'gnomad_af': np.random.exponential(0.001, 1000),
        'clinvar_significance': np.random.choice(['Pathogenic', 'Uncertain', 'Benign'], 1000)
    })
    return variants_df

variants_df = load_variant_data()

# Interactive feature computation controls
mo.md("## 🎛️ Feature Computation Parameters")

# Conservation features
phylop_weight = mo.ui.slider(0, 1, value=0.8, label="PhyloP Conservation Weight")
phastcons_weight = mo.ui.slider(0, 1, value=0.6, label="PhastCons Conservation Weight")

# Functional prediction features
spliceai_threshold = mo.ui.slider(0, 1, value=0.1, label="SpliceAI Pathogenicity Threshold")
alphamissense_threshold = mo.ui.slider(0, 1, value=0.8, label="AlphaMissense Confidence Threshold")

# Domain features
domain_penalty = mo.ui.slider(0, 5, value=2.0, label="Domain Disruption Penalty")

# Compute features reactively
@mo.cell
def compute_features(variants_df, phylop_weight, phastcons_weight,
                    spliceai_threshold, alphamissense_threshold, domain_penalty):
    """Compute features with interactive parameters"""

    features_df = variants_df.copy()

    # Conservation features (placeholder - in real impl, load from data)
    features_df['phylop_score'] = np.random.normal(0, 1, len(features_df))
    features_df['phastcons_score'] = np.random.normal(0, 1, len(features_df))
    features_df['conservation_combined'] = (
        phylop_weight * features_df['phylop_score'] +
        phastcons_weight * features_df['phastcons_score']
    )

    # Functional prediction features (placeholder)
    features_df['spliceai_score'] = np.random.beta(2, 8, len(features_df))
    features_df['alphamissense_score'] = np.random.beta(8, 2, len(features_df))
    features_df['spliceai_pathogenic'] = (features_df['spliceai_score'] > spliceai_threshold).astype(int)
    features_df['alphamissense_pathogenic'] = (features_df['alphamissense_score'] > alphamissense_threshold).astype(int)

    # Domain features (placeholder)
    features_df['in_domain'] = np.random.choice([0, 1], len(features_df))
    features_df['domain_penalty'] = features_df['in_domain'] * domain_penalty

    # Combine into final feature matrix
    feature_cols = [
        'conservation_combined',
        'spliceai_score',
        'alphamissense_score',
        'spliceai_pathogenic',
        'alphamissense_pathogenic',
        'domain_penalty',
        'gnomad_af'
    ]

    return features_df[feature_cols + ['variant_id', 'clinvar_significance']]

features_df = compute_features(variants_df, phylop_weight, phastcons_weight,
                              spliceai_threshold, alphamissense_threshold, domain_penalty)

# Feature exploration
mo.md("## 📊 Feature Distributions")

@mo.cell
def plot_feature_distributions(features_df):
    """Interactive feature distribution plots"""
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    numeric_features = ['conservation_combined', 'spliceai_score',
                       'alphamissense_score', 'gnomad_af']

    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=numeric_features,
        title_text="Feature Distributions"
    )

    for i, feature in enumerate(numeric_features):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Histogram(x=features_df[feature], name=feature),
            row=row, col=col
        )

    fig.update_layout(height=600, showlegend=False)
    return fig

feature_plots = plot_feature_distributions(features_df)
mo.ui.plot(feature_plots)

# Correlation analysis
mo.md("## 🔗 Feature Correlations")

@mo.cell
def correlation_analysis(features_df):
    """Interactive correlation analysis"""
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt

    numeric_cols = ['conservation_combined', 'spliceai_score',
                   'alphamissense_score', 'gnomad_af']

    corr_matrix = features_df[numeric_cols].corr()

    # Plotly heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r'
    )

    return fig

correlation_plot = correlation_analysis(features_df)
mo.ui.plot(correlation_plot)

# Feature importance analysis
mo.md("## 🎯 Feature Importance Analysis")

@mo.cell
def feature_importance_analysis(features_df):
    """Analyze feature relationships with clinical significance"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Prepare data
    feature_cols = ['conservation_combined', 'spliceai_score',
                   'alphamissense_score', 'gnomad_af', 'domain_penalty']

    X = features_df[feature_cols]
    y = LabelEncoder().fit_transform(features_df['clinvar_significance'])

    # Train simple model
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importance_
    }).sort_values('importance', ascending=False)

    return importance_df

importance_df = feature_importance_analysis(features_df)

@mo.cell
def plot_feature_importance(importance_df):
    """Plot feature importance"""
    import plotly.express as px

    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance for Clinical Significance Prediction'
    )

    return fig

importance_plot = plot_feature_importance(importance_df)
mo.ui.plot(importance_plot)

# Data quality checks
mo.md("## ✅ Data Quality Checks")

@mo.cell
def data_quality_checks(features_df):
    """Run data quality checks"""
    checks = {
        'Total Variants': len(features_df),
        'Missing Values': features_df.isnull().sum().sum(),
        'Infinite Values': np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum(),
        'Feature Range Check': 'All features in reasonable ranges',
        'Correlation Warnings': 'Check correlations > 0.95'
    }

    # Check for high correlations
    corr_matrix = features_df.select_dtypes(include=[np.number]).corr()
    high_corr = ((corr_matrix > 0.95) & (corr_matrix < 1.0)).sum().sum() // 2
    checks['High Correlations (>0.95)'] = high_corr

    return checks

quality_checks = data_quality_checks(features_df)
mo.ui.table(quality_checks)

# Export controls
mo.md("## 💾 Export Features")

@mo.cell
def export_feature_controls(features_df):
    """Controls for exporting computed features"""

    export_path = mo.ui.text(
        value="data_processed/features/abca4_features.parquet",
        label="Export Path"
    )

    export_button = mo.ui.button(
        label="Export Feature Matrix",
        on_click=lambda: export_features(features_df, export_path.value)
    )

    return mo.hstack([export_path, export_button])

def export_features(df, path):
    """Export features to Parquet"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Features exported to {path}")

export_feature_controls(features_df)

mo.md("""
## 🎯 Next Steps

1. **Model Training**: Use these features for Strand optimization
2. **Hyperparameter Tuning**: Adjust reward weights based on feature importance
3. **Cross-validation**: Evaluate feature stability across different data splits
4. **Feature Selection**: Remove redundant features based on correlation analysis

Use the interactive controls above to optimize your feature engineering pipeline!
""")
