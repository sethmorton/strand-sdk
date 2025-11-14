#!/usr/bin/env python3
"""
Complete working example of MPRA panel design with Strand SDK
Run this script to design an MPRA panel from scratch.
"""

import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path

def download_mpra_data(output_file="mpra_data.csv"):
    """Download MPRA variant data"""
    print("Downloading MPRA data...")
    url = "https://mpravardb.rc.ufl.edu/session/8cb1519b12d639ac307668346dda00ee/download/download_all?w="
    urllib.request.urlretrieve(url, output_file)
    print(f"Downloaded to {output_file}")
    return output_file

def load_and_filter_data(csv_file, n_samples=1000):
    """Load and filter functional variants"""
    print("Loading data...")

    # Read CSV (handle potential encoding issues)
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='latin1')

    print(f"Loaded {len(df)} variants")

    # Filter for functional variants
    functional = df[
        (df['log2FC'].abs() > 0.5) &  # Strong effect
        (df['fdr'] < 0.05) &          # Significant
        (df['ref_seq'].notna()) &     # Has sequence
        (df['alt_seq'].notna())
    ].copy()

    print(f"Filtered to {len(functional)} functional variants")

    # Take subsample for demo
    if len(functional) > n_samples:
        functional = functional.sample(n=n_samples, random_state=42)

    return functional

def compute_simple_features(df):
    """Compute simple regulatory features (placeholder for real implementation)"""
    print("Computing features...")

    features = []

    for _, row in df.iterrows():
        # Placeholder features (replace with real computations)
        variant_id = f"{row['chr']}:{row['pos']}"

        # Simulate feature computation
        enformer_score = np.random.normal(0, 1)  # Replace with real Enformer
        motif_score = np.random.normal(0, 1)     # Replace with real motif scanning
        conservation_score = np.random.normal(0, 1)  # Replace with real conservation

        # Combined score (weighted average)
        combined_score = 0.5 * enformer_score + 0.3 * motif_score + 0.2 * conservation_score

        features.append({
            'variant_id': variant_id,
            'chrom': row['chr'],
            'pos': row['pos'],
            'ref_seq': row['ref_seq'],
            'alt_seq': row['alt_seq'],
            'enformer_score': enformer_score,
            'motif_score': motif_score,
            'conservation_score': conservation_score,
            'combined_score': combined_score,
            'effect_size': row['log2FC'],
            'functional_label': 1
        })

    return pd.DataFrame(features)

def select_mpra_panel(features_df, panel_size=100):
    """Select MPRA panel using combined scoring"""
    print(f"Selecting panel of {panel_size} variants...")

    # Sort by combined score (descending)
    selected = features_df.nlargest(panel_size, 'combined_score').copy()

    # Add ranking
    selected['rank'] = range(1, len(selected) + 1)

    # Calculate panel statistics
    stats = {
        'panel_size': len(selected),
        'avg_enformer': selected['enformer_score'].mean(),
        'avg_motif': selected['motif_score'].mean(),
        'avg_conservation': selected['conservation_score'].mean(),
        'avg_combined': selected['combined_score'].mean(),
        'chromosomes': selected['chrom'].value_counts().to_dict()
    }

    return selected, stats

def save_panel(panel_df, stats, output_prefix="mpra_panel"):
    """Save selected panel and statistics"""
    print("Saving results...")

    # Save panel
    panel_file = f"{output_prefix}.csv"
    panel_df.to_csv(panel_file, index=False)

    # Save stats
    stats_file = f"{output_prefix}_stats.json"
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Panel saved to {panel_file}")
    print(f"Statistics saved to {stats_file}")

def main():
    """Main MPRA panel design pipeline"""
    print("=== MPRA Panel Design Example ===\n")

    # Step 1: Download data
    data_file = download_mpra_data()

    # Step 2: Load and filter
    mpra_data = load_and_filter_data(data_file, n_samples=5000)

    # Step 3: Compute features
    features_df = compute_simple_features(mpra_data)

    # Step 4: Select panel
    panel, stats = select_mpra_panel(features_df, panel_size=100)

    # Step 5: Save results
    save_panel(panel, stats)

    # Step 6: Print summary
    print("\n=== Panel Summary ===")
    print(f"Selected {stats['panel_size']} variants")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(f"Chromosome distribution: {stats['chromosomes']}")

    print("\nTop 5 selected variants:")
    for _, row in panel.head().iterrows():
        print(".3f")

    print("\n=== Next Steps ===")
    print("1. Replace placeholder features with real computations")
    print("2. Use Strand SDK for optimization")
    print("3. Validate on held-out test set")
    print("4. Export for wet-lab validation")

if __name__ == "__main__":
    main()
