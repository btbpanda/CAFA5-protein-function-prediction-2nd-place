#!/bin/bash
# pwd
conda --version

conda create --solver=libmamba -p $1/rapids-env -c rapidsai -c conda-forge -c nvidia  \
    rapids=23.02 python=3.8 cuda-version=11.2 -y

conda activate $1/rapids-env
which python
pip uninstall cupy numba -y # I reinstall default rapids cupy and numba via pypi due to the problems of my environment
# It is not actually needed in general case
pip install tqdm cupy-cuda112==10.6 numba==0.56.4 py-boost==0.4.3 