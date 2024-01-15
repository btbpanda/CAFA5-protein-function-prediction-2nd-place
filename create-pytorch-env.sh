#!/bin/bash

pwd

conda --version

conda create --solver=libmamba -p $1/pytorch-env python=3.9 -y
conda activate $1/pytorch-env
conda install --solver=libmamba pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install --solver=libmamba -c conda-forge cupy -y
pip install joblib tqdm pandas==1.3.5 pyyaml pyarrow numba==0.57.1 scikit-learn==1.0.2 numpy scipy fair-esm
pip install obonet pyvis transformers torchmetrics torchsummary sentencepiece psutil