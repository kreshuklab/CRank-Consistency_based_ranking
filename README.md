# CRank - Consistency based Ranking
### Ranking pre-trained segmentation models for zero-shot transferability 
([link to paper](https://arxiv.org/abs/2503.00450))

![Fig1](./figures/MICCAI_intro_figure.svg)


## Installation
#### Clone repository

Clone and navigate to this repository

```
git clone https://github.com/kreshuklab/CRank-Consistency_based_ranking.git
cd CRank-Consistency_based_ranking
```

#### Install model_ranking conda environment

```
conda env create -f environment.yaml
conda activate model_ranking
```


#### Download Pre-trained models

The model checkpoints and training configs are saved on zenodo and can be downloaded and unzipped https://doi.org/10.5281/zenodo.15209567


#### Use Weights and Biases logging
- Login/Sign up at https://wandb.ai/login
- Get your api token at https://wandb.ai/authorize (you'll be ask to provide this token on the first hylfm run, specifically on import wandb)
