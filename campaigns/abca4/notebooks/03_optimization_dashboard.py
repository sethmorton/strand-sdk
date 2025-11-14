#!/usr/bin/env python3
"""
ABCA4 Campaign - Optimization Dashboard

Interactive optimization dashboard for Strand campaigns.
Tune reward weights, visualize progress, and explore results in real-time.

Run with: marimo run notebooks/03_optimization_dashboard.py
Edit with: marimo edit notebooks/03_optimization_dashboard.py
"""

import marimo as mo
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

mo.md("# 🚀 ABCA4 Optimization Dashboard")

mo.md("""
This interactive dashboard lets you tune Strand optimization parameters
and visualize results in real-time. Adjust reward weights and see how they
affect the optimization trajectory.
""")

# Reward weight controls
mo.md("## ⚖️ Reward Weight Configuration")

enformer_weight = mo.ui.slider(0, 1, value=0.4, step=0.05, label="Enformer Δ Weight")
motif_weight = mo.ui.slider(0, 1, value=0.3, step=0.05, label="Motif Δ Weight")
conservation_weight = mo.ui.slider(0, 1, value=0.2, step=0.05, label="Conservation Weight")
dnafm_weight = mo.ui.slider(0, 1, value=0.1, step=0.05, label="DNA FM Δ Weight")

# Optimization parameters
panel_size = mo.ui.slider(50, 1000, value=100, step=50, label="Panel Size (K)")
num_iterations = mo.ui.slider(100, 10000, value=1000, step=100, label="Optimization Iterations")

# Reactive weight normalization
@mo.cell
def normalize_weights(enformer_weight, motif_weight, conservation_weight, dnafm_weight):
    """Normalize reward weights to sum to 1"""
    total = enformer_weight + motif_weight + conservation_weight + dnafm_weight
    if total == 0:
        return {'enformer': 0.25, 'motif': 0.25, 'conservation': 0.25, 'dnafm': 0.25}

    return {
        'enformer': enformer_weight / total,
        'motif': motif_weight / total,
        'conservation': conservation_weight / total,
        'dnafm': dnafm_weight / total
    }

normalized_weights = normalize_weights(enformer_weight, motif_weight, conservation_weight, dnafm_weight)

# Display normalized weights
mo.md("### Normalized Weights")
mo.ui.table({
    'Component': list(normalized_weights.keys()),
    'Weight': [f"{w:.3f}" for w in normalized_weights.values()]
})

# Simulate optimization (placeholder)
@mo.cell
def run_optimization_simulation(normalized_weights, panel_size, num_iterations):
    """Simulate optimization process (placeholder for real Strand optimization)"""

    # Simulate optimization progress
    iterations = np.arange(0, num_iterations, 50)
    best_scores = []

    for i in iterations:
        # Simulate score improvement over time
        noise = np.random.normal(0, 0.1)
        improvement = 1 - np.exp(-i / num_iterations)
        score = improvement + noise
        best_scores.append(max(0, min(1, score)))

    # Generate mock results
    results = {
        'iterations': iterations,
        'best_scores': best_scores,
        'final_score': best_scores[-1] if best_scores else 0,
        'convergence_iteration': np.where(np.array(best_scores) > 0.9)[0][0] if any(s > 0.9 for s in best_scores) else len(best_scores)
    }

    return results

optimization_results = run_optimization_simulation(normalized_weights, panel_size, num_iterations)

# Progress visualization
mo.md("## 📈 Optimization Progress")

@mo.cell
def plot_optimization_progress(optimization_results):
    """Plot optimization progress over time"""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=optimization_results['iterations'],
        y=optimization_results['best_scores'],
        mode='lines+markers',
        name='Best Score',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Optimization Progress',
        xaxis_title='Iteration',
        yaxis_title='Best Score',
        height=400
    )

    # Add convergence line
    fig.add_hline(
        y=0.9,
        line_dash="dash",
        line_color="red",
        annotation_text="Convergence Threshold"
    )

    return fig

progress_plot = plot_optimization_progress(optimization_results)
mo.ui.plot(progress_plot)

# Results summary
mo.md("## 📊 Optimization Results")

@mo.cell
def display_results_summary(optimization_results, panel_size):
    """Display optimization results summary"""
    summary = {
        'Panel Size (K)': panel_size,
        'Final Score': f"{optimization_results['final_score']:.3f}",
        'Convergence Iteration': optimization_results['convergence_iteration'] * 50,
        'Total Iterations': len(optimization_results['iterations']) * 50,
        'Convergence Rate': f"{(optimization_results['convergence_iteration'] / len(optimization_results['iterations'])):.1%}"
    }

    return summary

results_summary = display_results_summary(optimization_results, panel_size)
mo.ui.table(results_summary)

# Parameter sensitivity analysis
mo.md("## 🔍 Parameter Sensitivity")

@mo.cell
def sensitivity_analysis():
    """Analyze how different weight combinations affect final score"""

    # Generate parameter combinations
    weights_grid = np.array(np.meshgrid(
        np.linspace(0.1, 0.7, 4),  # enformer
        np.linspace(0.1, 0.7, 4),  # motif
        np.linspace(0.1, 0.3, 3),  # conservation
        np.linspace(0.0, 0.3, 3)   # dnafm
    )).T.reshape(-1, 4)

    # Normalize each combination
    normalized_grid = weights_grid / weights_grid.sum(axis=1, keepdims=True)

    # Simulate scores for each combination
    scores = []
    for weights in normalized_grid:
        # Simple scoring function based on weights
        score = (weights[0] * 0.8 + weights[1] * 0.7 +
                weights[2] * 0.6 + weights[3] * 0.5 +
                np.random.normal(0, 0.05))
        scores.append(max(0, min(1, score)))

    # Create results dataframe
    sensitivity_df = pd.DataFrame(normalized_grid, columns=['enformer', 'motif', 'conservation', 'dnafm'])
    sensitivity_df['score'] = scores

    return sensitivity_df.nlargest(10, 'score')

sensitivity_results = sensitivity_analysis()

mo.md("### Top 10 Parameter Combinations")
mo.ui.dataframe(sensitivity_results)

# Comparison with baselines
mo.md("## 🏁 Comparison with Baselines")

@mo.cell
def baseline_comparison():
    """Compare optimization results with baseline methods"""

    baselines = {
        'Random Selection': np.random.uniform(0.1, 0.3),
        'Conservation Only': 0.45,
        'Enformer Only': 0.52,
        'Motif Only': 0.38,
        'Strand (Current)': optimization_results['final_score']
    }

    return pd.DataFrame(list(baselines.items()), columns=['Method', 'Score'])

baseline_df = baseline_comparison()

@mo.cell
def plot_baseline_comparison(baseline_df):
    """Plot baseline comparison"""
    import plotly.express as px

    fig = px.bar(
        baseline_df,
        x='Method',
        y='Score',
        title='Optimization Method Comparison',
        color='Method'
    )

    fig.update_layout(height=400)
    return fig

baseline_plot = plot_baseline_comparison(baseline_df)
mo.ui.plot(baseline_plot)

# Export controls
mo.md("## 💾 Export Results")

@mo.cell
def export_optimization_results(optimization_results, normalized_weights):
    """Export optimization configuration and results"""

    export_config = {
        'weights': normalized_weights,
        'panel_size': panel_size,
        'iterations': num_iterations,
        'final_score': optimization_results['final_score'],
        'timestamp': pd.Timestamp.now().isoformat()
    }

    export_button = mo.ui.button(
        label="Export Configuration",
        on_click=lambda: print(f"Configuration: {export_config}")
    )

    return export_button

export_optimization_results(optimization_results, normalized_weights)

mo.md("""
## 🎯 Next Steps

1. **Real Optimization**: Replace simulation with actual Strand optimization
2. **Hyperparameter Search**: Use sensitivity analysis to guide parameter sweeps
3. **Cross-Validation**: Test optimization stability across different data splits
4. **Production Deployment**: Deploy optimized parameters for full campaign

Use the interactive controls to find the best reward weight configuration!
""")
