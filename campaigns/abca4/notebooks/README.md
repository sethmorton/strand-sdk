# ABCA4 Campaign - Marimo Notebooks

This directory contains interactive Marimo notebooks for the ABCA4 variant triage campaign. These notebooks provide reactive, reproducible analysis environments for data exploration, feature engineering, and optimization.

## 🚀 Quick Start

```bash
# Install Marimo and all interactive dependencies
uv pip install -e .[interactive]

# Start exploring data
uv run marimo edit notebooks/01_data_exploration.py

# Run as interactive web app
uv run marimo run notebooks/01_data_exploration.py

# Execute as regular Python scriptFEATURE_MATRIX
uv run python notebooks/01_data_exploration.py
```

## 📓 Notebook Overview

### 01_data_exploration.py
**Interactive ClinVar/gnomAD Data Exploration**
- SQL queries on variant dataframes
- Reactive filtering by gene, clinical significance, allele frequency
- Real-time data visualization and statistics
- Interactive dataframe exploration

**Key Features:**
- Dropdown filters for data sources and clinical significance
- Slider controls for allele frequency thresholds
- SQL queries: `SELECT * FROM clinvar_df WHERE gene LIKE '%ABCA4%'`
- Interactive plots and summary statistics

### 02_feature_engineering.py
**Real-time Feature Engineering Dashboard**
- Interactive parameter tuning for feature computation
- Live visualization of feature distributions and correlations
- Feature importance analysis for clinical significance prediction
- Data quality checks and export controls

**Key Features:**
- Sliders for conservation weights, splice thresholds, domain penalties
- Correlation heatmaps and distribution plots
- Random Forest feature importance analysis
- Export to Parquet format

### 03_optimization_dashboard.py
**Interactive Strand Optimization Tuning**
- Real-time reward weight adjustment
- Live optimization progress visualization
- Parameter sensitivity analysis
- Baseline method comparisons

**Key Features:**
- Reactive weight normalization (automatically sums to 1.0)
- Optimization progress plots with convergence tracking
- Sensitivity analysis across parameter combinations
- Comparison with random/conservation-only/enformer-only baselines

## 🔧 Marimo Advantages for Genomics

### Reactive Programming
- Change a parameter → instantly see updated results
- No manual "Run All Cells" required
- Deterministic execution prevents hidden state issues

### SQL Integration
```python
# Query dataframes with SQL
results = mo.sql("""
    SELECT gene, clnsig, af
    FROM clinvar_df
    WHERE af < 0.01 AND gene = 'ABCA4'
""")
```

### Interactive UI Elements
```python
# Reactive controls
gene_filter = mo.ui.dropdown(["ABCA4", "CFTR", "BRCA1"])
threshold = mo.ui.slider(0, 1, value=0.01)

# Automatically updates when controls change
```

### Git-Friendly
- Pure Python files (unlike Jupyter's JSON)
- Full version control and diff support
- Code review friendly

### Deployable Apps
- `marimo run notebook.py` creates interactive web apps
- Share results with non-technical stakeholders
- No Python installation required for viewers

## 🎯 Workflow Integration

These notebooks complement our `invoke` task automation:

1. **Data Download**: `invoke download-data` (automated scripts)
2. **Data Exploration**: `marimo edit notebooks/01_data_exploration.py` (interactive)
3. **Feature Engineering**: `marimo edit notebooks/02_feature_engineering.py` (interactive)
4. **Optimization**: `invoke run-optimization` + `marimo edit notebooks/03_optimization_dashboard.py` (hybrid)
5. **Reporting**: `invoke generate-report` (automated) + `marimo run notebooks/03_optimization_dashboard.py` (interactive dashboard)

## 🔄 Converting Existing Jupyter Notebooks

```bash
# Convert Jupyter notebook to Marimo
marimo convert old_notebook.ipynb > new_notebook.py

# Then edit interactively
marimo edit new_notebook.py
```

## 📚 Marimo Resources

- [Official Documentation](https://docs.marimo.io/)
- [GitHub Repository](https://github.com/marimo-team/marimo)
- [Interactive Playground](https://marimo.app/)

## 🎯 Next Steps

1. Install Marimo: `uv sync` (marimo is now in main dependencies)
2. Explore the data: `uv run marimo edit campaigns/abca4/notebooks/01_data_exploration.py`
3. Start feature engineering: `uv run marimo edit campaigns/abca4/notebooks/02_feature_engineering.py`
4. Tune optimization: `uv run marimo edit campaigns/abca4/notebooks/03_optimization_dashboard.py`

These notebooks provide the interactive foundation for reproducible, collaborative genomics research! 🔬✨
