# Current Results

```bash
TORCh after numpy squeeze :  0.27622056007385254
JAX after numpy conversion :  0.2762213945388794
computing ....
TORCH Input:  0.27622056007385254
TORCH After conv : -0.0045979442074894905
TORCH After group norm : -0.8400606513023376
TORCH After activation : 0.54152512550354
TORCH After max_pool : 0.7923572659492493
TORCH After embedder: torch.Size([1, 64, 32, 32])
TORCH After block 0: 0.8905758857727051
TORCH before skip : 0.8905758857727051
TORCH after conv proj skip : -0.38904091715812683
TORCH after skip : -0.6912142038345337
TORCH After block 1: 0.8190929889678955
TORCH before skip : 0.8190929889678955
TORCH after conv proj skip : -0.7010910511016846
TORCH after skip : -1.0914855003356934
TORCH After block 2: 0.4366658926010132
TORCH before skip : 0.4366658926010132
TORCH after conv proj skip : -0.2274589240550995
TORCH after skip : -2.7387194633483887
TORCH After block 3: 0.939188539981842
JAX Input:  0.2762213945388794
JAX After conv: -0.004597952589392662
JAX After group norm: -0.8400599360466003
JAX After activation: 0.5415254831314087
JAX After max_pool : 0.7923576831817627
JAX After embedder: (1, 32, 32, 64)
JAX After block 0: 0.8905760645866394
JAX before skip : 0.8905760645866394
JAX after conj proj skip : -0.38904136419296265
JAX after skip : -0.691214382648468
JAX After block 1: 0.8367286920547485
JAX before skip : 0.8367286920547485
JAX after conj proj skip : -0.6246893405914307
JAX after skip : -1.0879442691802979
JAX After block 2: 0.4456843137741089
JAX before skip : 0.4456843137741089
JAX after conj proj skip : -0.23142705857753754
JAX after skip : -2.8726773262023926
JAX After block 3: 0.8211888074874878

=== Benchmark Results ===

Timing Comparison:
PyTorch mean inference time: 7.62 ms
JAX mean inference time: 299.66 ms

Feature Similarity Metrics:
Mean MSE: 11.64722729
Mean Cosine Similarity: 0.35264814
Mean Maximum Absolute Difference: 42.28022766

Analysis:
⚠ Features show some differences
⚠ Notable maximum differences detected: 42.28022766

Speed Comparison: JAX is 0.03x slower than PyTorch
```

# ResNet 10

This project provides tools to convert JAX-based ResNet-10 weights from the [HIL-SERL](https://github.com/rail-berkeley/hil-serl) implementation to PyTorch format. The converted model is compatible with the Hugging Face Transformers library.

## Features

- Convert ResNet-10 weights from JAX to PyTorch format
- Automatic download of pretrained weights
- Integration with Hugging Face Hub
- Type-checked and well-tested codebase

## Usage

The model is available on Hugging Face Hub. You can use it as follows:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("helper2424/resnet10-imagenet-1k", trust_remote_code=True)
```

### Installation

1. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/helper2424/resnet-10.git
   cd resnet_10
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```


### How to convert

```bash
poetry run python convert_jax_to_pytorch.py --model_name helper2424/resnet10-imagenet-1k --push_to_hub True
```

### Validation

This script will download the model from the hub and validate that it works as expected.

```bash
poetry run python resnet_10/validate.py --model_name helper2424/resnet10-imagenet-1k
```
