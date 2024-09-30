# Multilinear Mixture of Experts:<br> Scalable Expert Specialization through Factorization

[![arXiv](https://img.shields.io/badge/arXiv-2402.12550-red)](https://arxiv.org/abs/2402.12550) [![project_page](https://img.shields.io/badge/project_page-orange)](https://james-oldfield.github.io/muMoE/)

## Abstract

> **Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization**<br>
James Oldfield, Markos Georgopoulos, Grigorios G. Chrysos, Christos Tzelepis, Yannis Panagakis, Mihalis A. Nicolaou, Jiankang Deng, Ioannis Patras<br>
*NeurIPS*, 2024 <br>
https://arxiv.org/abs/2402.12550 <br>
> **Abstract**: The Mixture of Experts (MoE) paradigm provides a powerful way to decompose dense layers into smaller, modular computations often more amenable to human interpretation, debugging, and editability. However, a major challenge lies in the computational cost of scaling the number of experts high enough to achieve fine-grained specialization. In this paper, we propose the Multilinear Mixture of Experts (μMoE) layer to address this, focusing on vision models. μMoE layers enable scalable expert specialization by performing an implicit computation on prohibitively large weight tensors entirely in factorized form. Consequently, μMoEs (1) avoid the restrictively high inference-time costs of dense MoEs, yet (2) do not inherit the training issues of the popular sparse MoEs' discrete (non-differentiable) expert routing. We present both qualitative and quantitative evidence that scaling μMoE layers when fine-tuning foundation models for vision tasks leads to more specialized experts at the class-level, further enabling manual bias correction in CelebA attribute classification. Finally, we show qualitative results demonstrating the expert specialism achieved when pre-training large GPT2 and MLP-Mixer models with parameter-matched μMoE blocks at every layer, maintaining comparable accuracy.

### The μMoE forward pass
<img src="./images/anim.gif" width="400"/>

> The forward pass of an (unfactorized) μMoE layer as a series of two tensor contractions: the experts' weight matrices are matrix-multiplied with the input vector and summed (weighted by the expert coefficients).


## Install

First, please install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### CPμMoE

```python
# initialise a rank-R CPMMoE
import torch
from MuMoE import CPMuMoE

n_experts = 32; batch_size = 4; in_dim = 4; out_dim = 8
x = torch.randn(batch_size, in_dim)

R = 32
model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=CPMuMoE, expert_dims=[n_experts], ranks=R)
y = model(x)

# or with two levels of hierarchy
model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=CPMuMoE, expert_dims=[n_experts, 2], ranks=R, hierarchy=2)
y = model(x)
```
### TRμMoE

```python
import torch
from MuMoE import TRMuMoE

n_experts = 32; batch_size = 4; in_dim = 4; out_dim = 8
x = torch.randn(batch_size, in_dim)

r1 = 4 ; r2 = 4 ; r3 = 4  # TR
ranks = [[r1, n_experts, r2], [r2, in_dim, r3], [r3, out_dim, r1]]
model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=TRMuMoE, expert_dims=[n_experts], ranks=ranks)
y = model(x)

# or with two levels of hierarchy
r1 = 4 ; r2 = 4 ; r3 = 4 ; r4 = 4  # TR
ranks = [[r1, n_experts, r2], [r2, n_experts, r3], [r3, in_dim, r4], [r4, out_dim, r1]]
model = MuMoE(input_dim=in_dim, output_dim=out_dim, MuMoE_layer=TRMuMoE, expert_dims=[n_experts, 2], ranks=ranks, hierarchy=2)
y = model(x)
```

## Experiments


### MLP-Mixer

To reproduce the results for the baseline MLP mixer s16 model for the full 300 epochs, we use 4x80GB A100 GPUs. Please run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision "bf16" mlp_mixer.py --layer "MLP" --stochastic_depth
```

And for the μMoE models:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision "bf16" mlp_mixer.py --layer "CPMuMoE" --stochastic_depth
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision "bf16" mlp_mixer.py --layer "TRMuMoE" --stochastic_depth
```

### GPT-2 pre-training on openwebtext2

Please see the `./experiments/nanoGPT` directory. To reproduce the results reported, please run:

The baseline:

```bash
torchrun --standalone --nproc_per_node=4 train.py --layer_type='linear' --compile=False config/train_gpt2.py
```

And for the μMoE models:

```bash
torchrun --standalone --nproc_per_node=4 train.py --layer_type='CPMuMoE' --compile=False config/train_gpt2.py --n_experts=256
torchrun --standalone --nproc_per_node=4 train.py --layer_type='TRMuMoE' --compile=False config/train_gpt2.py --n_experts=256
```

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{oldfield2024mumoe,
    title={Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization},
    author={James Oldfield and Markos Georgopoulos and Grigorios G. Chrysos and Christos Tzelepis and Yannis Panagakis and Mihalis A. Nicolaou and Jiankang Deng and Ioannis Patras},
    booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=bIa03mAtxQ}
}
```

## Contact

**Please feel free to get in touch at**: `j.a.oldfield@qmul.ac.uk`
