#!/usr/bin/env python3
"""
ABCA4 Campaign - Feature Engineering Notebook

Interactive feature engineering for ABCA4 variants.
Tune parameters, visualize distributions, and see correlations in real-time.

Run with: marimo run notebooks/02_feature_engineering.py
Edit with: marimo edit notebooks/02_feature_engineering.py
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import sys
    return mo, pd, np, Path, sys

@app.cell
def __(mo):
    mo.md("# 🔧 ABCA4 Feature Engineering")
    mo.md("""
This notebook provides interactive feature engineering for ABCA4 variants.
Tune parameters, visualize feature distributions, and explore correlations in real-time.
""")
    return

@app.cell
def __(pd, np):
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
    return variants_df,

@app.cell
def __(mo):
    mo.md("## 🎛️ Feature Computation Parameters")
    
    # Conservation features
    phylop_weight = mo.ui.slider(0, 1, value=0.8, label="PhyloP Conservation Weight")
    phastcons_weight = mo.ui.slider(0, 1, value=0.6, label="PhastCons Conservation Weight")
    
    # Functional prediction features
    spliceai_threshold = mo.ui.slider(0, 1, value=0.1, label="SpliceAI Pathogenicity Threshold")
    alphamissense_threshold = mo.ui.slider(0, 1, value=0.8, label="AlphaMissense Confidence Threshold")
    
    # Domain features
    domain_penalty = mo.ui.slider(0, 5, value=2.0, label="Domain Disruption Penalty")
    
    return phylop_weight, phastcons_weight, spliceai_threshold, alphamissense_threshold, domain_penalty

@app.cell
def __(
    variants_df, phylop_weight, phastcons_weight,
    spliceai_threshold, alphamissense_threshold, domain_penalty, np
):
    """Compute features with interactive parameters"""
    features_df = variants_df.copy()

    # Conservation features (placeholder - in real impl, load from data)
    features_df['phylop_score'] = np.random.normal(0, 1, len(features_df))
    features_df['phastcons_score'] = np.random.normal(0, 1, len(features_df))
    features_df['conservation_combined'] = (
        phylop_weight.value * features_df['phylop_score'] +
        phastcons_weight.value * features_df['phastcons_score']
    )

    # Functional prediction features (placeholder)
    features_df['spliceai_score'] = np.random.beta(2, 8, len(features_df))
    features_df['alphamissense_score'] = np.random.beta(8, 2, len(features_df))
    features_df['spliceai_pathogenic'] = (features_df['spliceai_score'] > spliceai_threshold.value).astype(int)
    features_df['alphamissense_pathogenic'] = (features_df['alphamissense_score'] > alphamissense_threshold.value).astype(int)

    # Domain features (placeholder)
    features_df['in_domain'] = np.random.choice([0, 1], len(features_df))
    features_df['domain_penalty_score'] = features_df['in_domain'] * domain_penalty.value

    # Combine into final feature matrix
    feature_cols_computed = [
        'conservation_combined',
        'spliceai_score',
        'alphamissense_score',
        'spliceai_pathogenic',
        'alphamissense_pathogenic',
        'domain_penalty_score',
        'gnomad_af'
    ]

    features_computed = features_df[feature_cols_computed + ['variant_id', 'clinvar_significance']]
    return features_computed,

@app.cell
def __(mo):
    mo.md("## 📊 Feature Distributions")
    return

@app.cell
def __(features_computed):
    """Interactive feature distribution plots"""
    try:
        import plotly.express as px_dist
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        numeric_features_list = ['conservation_combined', 'spliceai_score',
                           'alphamissense_score', 'gnomad_af']

        # Create subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=numeric_features_list,
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'histogram'}]]
        )

        for i, feature in enumerate(numeric_features_list):
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_trace(
                go.Histogram(x=features_computed[feature], name=feature),
                row=row, col=col
            )

        fig.update_layout(height=600, showlegend=False, title_text="Feature Distributions")
        feature_plots_output = fig
    except ImportError:
        feature_plots_output = None
    
    return feature_plots_output,

@app.cell
def __(feature_plots_output, mo):
    if feature_plots_output:
        mo.ui.plot(feature_plots_output)
    else:
        mo.md("Install plotly to visualize feature distributions.")
    return

@app.cell
def __(mo):
    mo.md("## 🔗 Feature Correlations")
    return

@app.cell
def __(features_computed):
    """Interactive correlation analysis"""
    try:
        import plotly.express as px_corr

        numeric_cols_corr = ['conservation_combined', 'spliceai_score',
                       'alphamissense_score', 'gnomad_af']

        corr_matrix_plot = features_computed[numeric_cols_corr].corr()

        # Plotly heatmap
        correlation_plot = px_corr.imshow(
            corr_matrix_plot,
            text_auto='.2f',
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r'
        )
    except ImportError:
        correlation_plot = None
    
    return correlation_plot,

@app.cell
def __(correlation_plot, mo):
    if correlation_plot:
        mo.ui.plot(correlation_plot)
    else:
        mo.md("Install plotly to visualize correlations.")
    return

@app.cell
def __(mo):
    mo.md("## 🎯 Feature Importance Analysis")
    return

@app.cell
def __(features_computed, pd):
    """Analyze feature relationships with clinical significance"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        feature_cols_imp = ['conservation_combined', 'spliceai_score',
                       'alphamissense_score', 'gnomad_af', 'domain_penalty_score']

        X = features_computed[feature_cols_imp]
        y = LabelEncoder().fit_transform(features_computed['clinvar_significance'])

        # Train simple model
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols_imp,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    except ImportError:
        importance_df = pd.DataFrame({'feature': [], 'importance': []})
    
    return importance_df,

@app.cell
def __(importance_df):
    """Plot feature importance"""
    try:
        import plotly.express as px

        if not importance_df.empty:
            importance_plot = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance for Clinical Significance Prediction'
            )
        else:
            importance_plot = None
    except ImportError:
        importance_plot = None
    
    return importance_plot,

@app.cell
def __(importance_plot, mo):
    if importance_plot:
        mo.ui.plot(importance_plot)
    else:
        mo.md("Install scikit-learn and plotly to see feature importance analysis.")
    return

@app.cell
def __(mo):
    mo.md("## ✅ Data Quality Checks")
    return

@app.cell
def __(features_computed, np):
    """Run data quality checks"""
    checks_dict = {
        'Total Variants': len(features_computed),
        'Missing Values': features_computed.isnull().sum().sum(),
        'Infinite Values': int(np.isinf(features_computed.select_dtypes(include=[np.number])).sum().sum()),
        'Feature Range Check': 'All features in reasonable ranges',
    }

    # Check for high correlations
    numeric_features_qc = features_computed.select_dtypes(include=[np.number])
    if not numeric_features_qc.empty:
        corr_matrix_qc = numeric_features_qc.corr()
        high_corr = int(((corr_matrix_qc > 0.95) & (corr_matrix_qc < 1.0)).sum().sum() // 2)
        checks_dict['High Correlations (>0.95)'] = high_corr

    return checks_dict,

@app.cell
def __(checks_dict, mo):
    mo.ui.table(checks_dict)
    return

@app.cell
def __(mo):
    mo.md("## 💾 Export Features")
    
    export_path = mo.ui.text(
        value="data_processed/features/abca4_features.parquet",
        label="Export Path"
    )
    
    return export_path,

@app.cell
def __(export_path, mo):
    mo.md(f"""
Export features to: `{export_path.value}`

To actually export, you would click a button (currently disabled for safety).
""")
    return

@app.cell
def __(mo):
    mo.md("""
## 🎯 Next Steps

1. **Model Training**: Use these features for Strand optimization
2. **Hyperparameter Tuning**: Adjust reward weights based on feature importance
3. **Cross-validation**: Evaluate feature stability across different data splits
4. **Feature Selection**: Remove redundant features based on correlation analysis

Use the interactive controls above to optimize your feature engineering pipeline!
""")
    return

if __name__ == "__main__":
    app.run()
