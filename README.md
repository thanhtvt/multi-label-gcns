<div align="center">

# Multi-label Image Recognition with GCN and its Variants

<a href="https://github.com/pre-commit/pre-commit"><img alt="Python" src="https://img.shields.io/badge/-Python_3.6+-blue?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

<a href="https://arxiv.org/abs/1904.03582"><img alt="Paper" src="http://img.shields.io/badge/paper-arxiv.1904.03582-B31B1B.svg"></a>
<a href="https://openaccess.thecvf.com/CVPR2019"><img alt="Conference" src="https://img.shields.io/badge/CVPR-2019-4b44ce.svg"></a>

</div>

## Overview
This repo contains an implementation (and a few variants) of the paper [Multi-label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582). This repo is following [Lightning Hydra](https://github.com/ashleve/lightning-hydra-template) template.

In general, the model is the combination of a CNN-based as the image representation extractor and a GCN-based as the label embedding. Figure 1 describes architecture of the model.

<div align="center">
<img src="./static/architecture.png" alt="architecture"/>

Figure 1. The overall architecture of the model.
</div>

## Dataset

Currently we have ready-to-use [VOCDectection](https://pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html) pre-processor. More datasets will be added soon.

## Installation
You should have Python 3.7 or higher. I highly recommend creating a virual environment like venv or [Conda](https://docs.conda.io/en/latest/miniconda.html). For example:

```bash
# clone project
git clone https://github.com/thanhtvt/multi-label-gcns.git
cd multi-label-gcns

# [OPTIONAL] create conda environment
conda create -n mlgcn python=3.8
conda activate mlgcn

# install requirements
pip install -r requirements.txt
```

## 🚀 Quick start
To train model with default configuration, going to `src` folder and run:

```bash
# train on CPU
python train.py trainer=cpu

# train on GPU
python train.py trainer=gpu
```

To train model with chosen experiment configuration from `configs/experiment` folder, run:

```bash
python train.py experiment=multi-label_base
```

To override any parameter from commandline, run:

```bash
python train.py logger=csv trainer.max_epochs=100
```
