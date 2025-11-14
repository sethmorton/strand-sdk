# MPRA Implementation Guide

How to actually implement MPRA panel design in Strand SDK.

## CLI Usage

### Basic Panel Selection
```bash
# Select 100 variants using combined features
strand mpra select --dataset mpra_data.csv --panel-size 100 --output selected_panel.csv

# Use specific features
strand mpra select --features enformer,motif,conservation --weights 0.5,0.3,0.2
```

### Optimization Campaign
```bash
# Run CMA-ES optimization for 100 iterations
strand mpra optimize --dataset mpra_data.csv --method cmaes --iterations 100 --panel-size 50

# Use custom reward function
strand mpra optimize --reward my_reward.py --output optimized_panel.json
```

## Python API

### Simple Selection
```python
from strand_mpra import MPRAPanelSelector

# Load data
selector = MPRAPanelSelector.from_csv("mpra_data.csv")

# Select panel
panel = selector.select_top_k(k=100, features=['enformer', 'motif', 'conservation'])
panel.save("my_panel.csv")
```

### Custom Reward Function
```python
from strand.rewards.base import RewardBlock

class MyMPRAReward(RewardBlock):
    def compute_reward(self, sequence_context):
        # Your custom logic here
        enformer_score = compute_enformer(sequence_context)
        motif_score = compute_motif(sequence_context)
        return enformer_score * 0.6 + motif_score * 0.4

# Use in campaign
from strand.campaigns import OptimizationCampaign

campaign = OptimizationCampaign(reward=MyMPRAReward(), panel_size=100)
result = campaign.run()
```

## Configuration

### YAML Config
```yaml
mpra:
  dataset: "data/mpra_variants.csv"
  panel_size: 100
  features:
    - name: enformer
      weight: 0.5
      model: "EleutherAI/enformer-official-rough"
    - name: motif
      weight: 0.3
      tf_list: ["CTCF", "MYC", "TP53"]
    - name: conservation
      weight: 0.2
      track: "phyloP100way.bw"

optimization:
  method: "cmaes"
  iterations: 100
  sigma: 0.5
```

### Environment Setup
```bash
# Install dependencies
pip install strand-sdk enformer-pytorch pybigwig MOODS-python pyjaspar

# Download required data
wget https://mpravardb.rc.ufl.edu/mpra_data.csv
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw
```

## File Formats

### Input Dataset
```csv
variant_id,chrom,start,end,ref_seq,alt_seq,effect_size,functional_label
var_1,chr1,1000000,1000000,ATCGATCG...,ATCGATCC...,1.2,1
var_2,chr2,500000,500000,GCTAGCTA...,GCTAGCCA...,-0.8,0
```

### Output Panel
```csv
variant_id,score,rank,features
var_1,2.34,1,enformer:1.2,motif:0.8,conservation:0.34
var_2,2.12,2,enformer:1.1,motif:0.7,conservation:0.32
```

## Troubleshooting

### Common Issues

**Out of memory**: Use batch processing
```python
selector = MPRAPanelSelector(batch_size=50)  # Process in batches
```

**Slow computation**: Enable GPU
```python
import torch
torch.set_default_device('cuda')  # Use GPU
```

**No motifs found**: Check TF names
```python
from pyjaspar import jaspardb
jdb = jaspardb()
available = [m.name for m in jdb.fetch_motifs_by_species('Homo sapiens')[:10]]
print("Available TFs:", available)
```

### Performance Tips

1. **Cache features**: Pre-compute and save feature matrices
2. **Use GPU**: Move models to CUDA when available
3. **Batch processing**: Don't load everything into memory
4. **Parallel**: Use multiple cores for independent computations

## Testing

### Unit Tests
```python
def test_mpra_selection():
    # Create test data
    test_variants = create_test_mpra_data(n=100)

    # Test selection
    selector = MPRAPanelSelector(test_variants)
    panel = selector.select_top_k(k=10)

    assert len(panel) == 10
    assert all(v.functional_label for v in panel)  # Should select functional variants

if __name__ == "__main__":
    test_mpra_selection()
    print("All tests passed!")
```

### Integration Tests
```python
def test_full_pipeline():
    """Test complete MPRA pipeline"""

    # Download data
    data = download_mpra_data()

    # Compute features
    features = compute_all_features(data)

    # Select panel
    panel = select_panel(features, k=50)

    # Validate enrichment
    enrichment = calculate_enrichment(panel, data)
    assert enrichment > 2.0  # Should be enriched

    print(f"Pipeline test passed! Enrichment: {enrichment:.2f}x")
```

## Deployment

### Docker Container
```dockerfile
FROM python:3.9

# Install dependencies
RUN pip install strand-sdk enformer-pytorch pybigwig MOODS-python pyjaspar

# Copy code
COPY . /app
WORKDIR /app

# Download data
RUN wget -O /data/mpra_data.csv https://mpravardb.rc.ufl.edu/mpra_data.csv

CMD ["python", "run_mpra_pipeline.py"]
```

### Cloud Deployment
```python
# AWS Batch or similar
import boto3

def run_mpra_on_cloud(dataset_url, panel_size=100):
    # Submit job
    client = boto3.client('batch')

    response = client.submit_job(
        jobName='mpra-panel-design',
        jobQueue='mpra-queue',
        jobDefinition='mpra-job-definition',
        parameters={
            'dataset_url': dataset_url,
            'panel_size': str(panel_size)
        }
    )

    return response['jobId']
```

This guide shows you exactly how to implement MPRA panel design without getting lost in theory.
