#!/usr/bin/env python3
"""
ABCA4 Campaign - Optimization Dashboard

Interactive optimization dashboard for Strand campaigns.
Tune reward weights, visualize progress, and explore results in real-time.

Run with: marimo run notebooks/03_optimization_dashboard.py
Edit with: marimo edit notebooks/03_optimization_dashboard.py
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
    return mo, pd, np, Path

@app.cell
def __(mo):
    mo.md("# 🚀 ABCA4 Optimization Dashboard")
    mo.md("""
This interactive dashboard lets you tune Strand optimization parameters
and visualize results in real-time. Adjust reward weights and see how they
affect the optimization trajectory.
""")
    return

@app.cell
def __(mo):
    mo.md("## ⚖️ Reward Weight Configuration")
    
    enformer_weight = mo.ui.slider(0, 1, value=0.4, step=0.05, label="Enformer Δ Weight")
    motif_weight = mo.ui.slider(0, 1, value=0.3, step=0.05, label="Motif Δ Weight")
    conservation_weight = mo.ui.slider(0, 1, value=0.2, step=0.05, label="Conservation Weight")
    dnafm_weight = mo.ui.slider(0, 1, value=0.1, step=0.05, label="DNA FM Δ Weight")
    
    # Optimization parameters
    panel_size = mo.ui.slider(50, 1000, value=100, step=50, label="Panel Size (K)")
    num_iterations = mo.ui.slider(100, 10000, value=1000, step=100, label="Optimization Iterations")
    
    return enformer_weight, motif_weight, conservation_weight, dnafm_weight, panel_size, num_iterations

@app.cell
def __(enformer_weight, motif_weight, conservation_weight, dnafm_weight):
    """Normalize reward weights to sum to 1"""
    total = enformer_weight.value + motif_weight.value + conservation_weight.value + dnafm_weight.value
    if total == 0:
        normalized_weights = {'enformer': 0.25, 'motif': 0.25, 'conservation': 0.25, 'dnafm': 0.25}
    else:
        normalized_weights = {
            'enformer': enformer_weight.value / total,
            'motif': motif_weight.value / total,
            'conservation': conservation_weight.value / total,
            'dnafm': dnafm_weight.value / total
        }
    
    return normalized_weights,

@app.cell
def __(normalized_weights, mo):
    mo.md("### Normalized Weights")
    weights_table = {
        'Component': list(normalized_weights.keys()),
        'Weight': [f"{w:.3f}" for w in normalized_weights.values()]
    }
    mo.ui.table(weights_table)
    return

@app.cell
def __(normalized_weights, panel_size, num_iterations, np):
    """Simulate optimization process (placeholder for real Strand optimization)"""

    # Simulate optimization progress
    iterations = np.arange(0, num_iterations.value, max(1, num_iterations.value // 20))
    best_scores = []

    for i in iterations:
        # Simulate score improvement over time
        noise = np.random.normal(0, 0.1)
        improvement = 1 - np.exp(-i / max(1, num_iterations.value))
        score_opt = improvement + noise
        best_scores.append(max(0, min(1, score_opt)))

    # Generate mock results
    convergence_iter_indices = np.where(np.array(best_scores) > 0.9)[0]
    convergence_iteration = int(convergence_iter_indices[0]) if len(convergence_iter_indices) > 0 else len(best_scores)
    
    optimization_results = {
        'iterations': iterations,
        'best_scores': best_scores,
        'final_score': best_scores[-1] if best_scores else 0,
        'convergence_iteration': convergence_iteration
    }

    return optimization_results,

@app.cell
def __(mo):
    mo.md("## 📈 Optimization Progress")
    return

@app.cell
def __(optimization_results):
    """Plot optimization progress over time"""
    try:
        import plotly.graph_objects as go

        fig_progress = go.Figure()

        fig_progress.add_trace(go.Scatter(
            x=optimization_results['iterations'],
            y=optimization_results['best_scores'],
            mode='lines+markers',
            name='Best Score',
            line=dict(color='blue', width=2)
        ))

        fig_progress.update_layout(
            title='Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Best Score',
            height=400
        )

        # Add convergence line
        fig_progress.add_hline(
            y=0.9,
            line_dash="dash",
            line_color="red",
            annotation_text="Convergence Threshold"
        )

        progress_plot = fig_progress
    except ImportError:
        progress_plot = None

    return progress_plot,

@app.cell
def __(progress_plot, mo):
    if progress_plot:
        mo.ui.plot(progress_plot)
    else:
        mo.md("Install plotly to visualize optimization progress.")
    return

@app.cell
def __(mo):
    mo.md("## 📊 Optimization Results")
    return

@app.cell
def __(optimization_results, panel_size):
    """Display optimization results summary"""
    summary_dict = {
        'Panel Size (K)': panel_size.value,
        'Final Score': f"{optimization_results['final_score']:.3f}",
        'Convergence Iteration': int(optimization_results['convergence_iteration'] * (optimization_results['iterations'][1] - optimization_results['iterations'][0]) if len(optimization_results['iterations']) > 1 else 0),
        'Total Iterations': int(optimization_results['iterations'][-1] if len(optimization_results['iterations']) > 0 else 0),
        'Convergence Rate': f"{(optimization_results['convergence_iteration'] / max(1, len(optimization_results['iterations']))):.1%}"
    }

    return summary_dict,

@app.cell
def __(summary_dict, mo):
    mo.ui.table(summary_dict)
    return

@app.cell
def __(mo):
    mo.md("## 🔍 Parameter Sensitivity")
    return

@app.cell
def __(np, pd):
    """Analyze how different weight combinations affect final score"""

    # Generate parameter combinations
    weights_grid = np.array(np.meshgrid(
        np.linspace(0.1, 0.7, 4),  # enformer
        np.linspace(0.1, 0.7, 4),  # motif
        np.linspace(0.1, 0.3, 3),  # conservation
        np.linspace(0.0, 0.3, 3)   # dnafm
    )).T.reshape(-1, 4)

    # Normalize each combination
    normalized_grid = weights_grid / np.maximum(weights_grid.sum(axis=1, keepdims=True), 1e-10)

    # Simulate scores for each combination
    scores = []
    for weights in normalized_grid:
        # Simple scoring function based on weights
        score_sens = (weights[0] * 0.8 + weights[1] * 0.7 +
                weights[2] * 0.6 + weights[3] * 0.5 +
                np.random.normal(0, 0.05))
        scores.append(max(0, min(1, score_sens)))

    # Create results dataframe
    sensitivity_df = pd.DataFrame(normalized_grid, columns=['enformer', 'motif', 'conservation', 'dnafm'])
    sensitivity_df['score'] = scores

    sensitivity_results = sensitivity_df.nlargest(10, 'score')
    return sensitivity_results,

@app.cell
def __(sensitivity_results, mo):
    mo.md("### Top 10 Parameter Combinations")
    mo.ui.dataframe(sensitivity_results)
    return

@app.cell
def __(mo):
    mo.md("## 🏁 Comparison with Baselines")
    return

@app.cell
def __(optimization_results, np, pd):
    """Compare optimization results with baseline methods"""

    baselines = {
        'Random Selection': float(np.random.uniform(0.1, 0.3)),
        'Conservation Only': 0.45,
        'Enformer Only': 0.52,
        'Motif Only': 0.38,
        'Strand (Current)': float(optimization_results['final_score'])
    }

    baseline_df = pd.DataFrame(list(baselines.items()), columns=['Method', 'Score'])
    return baseline_df,

@app.cell
def __(baseline_df):
    """Plot baseline comparison"""
    try:
        import plotly.express as px_base

        fig_baseline = px_base.bar(
            baseline_df,
            x='Method',
            y='Score',
            title='Optimization Method Comparison',
            color='Method'
        )

        fig_baseline.update_layout(height=400)
        baseline_plot = fig_baseline
    except ImportError:
        baseline_plot = None

    return baseline_plot,

@app.cell
def __(baseline_plot, mo):
    if baseline_plot:
        mo.ui.plot(baseline_plot)
    else:
        mo.md("Install plotly to visualize baseline comparison.")
    return

@app.cell
def __(mo):
    mo.md("## 💾 Export Results")
    return

@app.cell
def __(normalized_weights, panel_size, num_iterations, optimization_results, pd):
    """Export optimization configuration and results"""

    export_config = {
        'weights': normalized_weights,
        'panel_size': panel_size.value,
        'iterations': num_iterations.value,
        'final_score': optimization_results['final_score'],
        'timestamp': pd.Timestamp.now().isoformat()
    }

    return export_config,

@app.cell
def __(export_config, mo):
    mo.md(f"""
**Export Configuration:**

```
{export_config}
```

To export, you would typically save this to a JSON or YAML file.
""")
    return

@app.cell
def __(mo):
    mo.md("""
## 🎯 Next Steps

1. **Real Optimization**: Replace simulation with actual Strand optimization
2. **Hyperparameter Search**: Use sensitivity analysis to guide parameter sweeps
3. **Cross-Validation**: Test optimization stability across different data splits
4. **Production Deployment**: Deploy optimized parameters for full campaign

Use the interactive controls to find the best reward weight configuration!
""")
    return

if __name__ == "__main__":
    app.run()
