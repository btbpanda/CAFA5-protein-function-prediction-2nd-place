Hello!

Here are the instructions to reproduce the CAFA5 2nd solution using given code

# Contents

nn_solution                 : scripts for training Neural Network base models
protlib                     : utils and code to train Py-Boost and LogReg models, data preprocessing and efficient metric computation
protnn                      : utils and code to train GCN stacker model
CAFA5PIpeline.ipynb         : CAFA5PIpeline.ipynb - notebook contains all the scripts calls and detailed explanation of each step. Also, contains directory structure (shoul be considered as both `directory_structure.txt` and `entry_points.md`)
Download.ipynb              : since produced artifacts are quite large, we consider to store it in the cloud storage, instead uploading it on Kagge. To download all trained models, please execute this notebook. Explanation of contents is also provided 
config.yaml                 : config used to execute training and inference. **Note!!** the artifacts will be stored for 6 month only. After that, you will need to compute it by yourself.
create-pytorch-env.sh       : install all the requirements to run all deep learning parts
create-rapids-env.sh        : install all the requirements to run processing and ML steps


# HARDWARE 

We used the following setup to train:

    - 24 CPUs
    - 512 GB RAM
    - 2 x Tesla V100 32 GB

Minimal required hardware:
    
    - 8 CPUs
    - 64 GB RAM
    - 1 x Tesla V100 32 GB    
    - 300 GB disk space
    
# Software

    - Ubuntu 18.04
    - Nvidia driver version 450 
    - default python 3.8 to run `CAFA5PIpeline.ipynb` and `Download.ipynb` notebooks. Only requred libraries are `pyyaml` to read `config.yaml' and `kaggle` to obtain the original dataset via API
    - conda >= 23.5.2. We need one of the latest version to use Mamba solver. Otherwise, setup the environments will take hours
    
Other required tools will be installed via `create-pytorch-env.sh` and `create-rapids-env.sh` scripts. 

    - pytorch-env is the environment to train DL models. It will install pytorch, cupy, and some extra BIO libraries
    - rapids-env is the enviromnent to do preprocessing and train ML models. It uses NVIDIA RAPIDS toolkit (cudf) and cupy libraries to make the efficient dataprocessing, metric computation (including custom CUDA kernels for graph manipulation) and custom ML algorithms implementations
    

# DATA AND ENV SETUP

To install default python dependencies, please execute `pip install -r requirements.txt`

To obtain the original Kaggle dataset, please execute (be sure you get personal access kaggle token)

```bash
kaggle competitions download -c cafa-5-protein-function-prediction
unzip cafa-5-protein-function-prediction.zip
```

# NEXT STEPS

To reproduce the solution, please step by step execute the notebook `CAFA5PIpeline.ipynb`

To download all the artifacts without running the code, please execute step by step `Download.ipynb`

Explanation is also provided for each step for both notebooks