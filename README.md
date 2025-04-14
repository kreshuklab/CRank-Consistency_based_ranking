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


## Running Scripts

Within the scripts folder there are mulitple scripts that allow easy interaction with the repository and to reproduce the results of the paper.

#### Activate conda environment

```
conda activate model_ranking
```

#### Batch run prediciton and evaluation
To perform a batch of inference runs between a set of source models transferred to a set of target datasets and then calculate ground-truth performance metrics and consistency scores, specifiy a suitable ","meta_config.yaml" file (example given meta_configs/) and run the script batch_prediction_evaluation.py. To run on GPU optionally specify a cuda device.

```
CUDA_VISIBLE_DEVICES=6 python batch_prediction_evaluation.py --config path/to/meta_config.yaml
```

#### Batch run only Evaluation/Consistency
To perform a batch of performance evaluations or consistency scores between a set of source models transferred to a set of target datasets, specify a suitbale "meta_config.yaml" with "run_mode=evaluation or consistency" and then run the script "batch_evaluate.py" or "batch_consistency.py" respectively.

```
CUDA_VISIBLE_DEVICES=6 python batch_evaluate.py --config path/to/evaluation/meta_config.yaml
```

```
CUDA_VISIBLE_DEVICES=6 python batch_consistency.py --config path/to/consistency/meta_config.yaml
```


#### Use Weights and Biases logging
- Login/Sign up at https://wandb.ai/login
- Get your api token at https://wandb.ai/authorize (you'll be ask to provide this token on the first run, specifically on import wandb)





