This repository contains the code and models for our paper:

**What Can Mamba Learn In-Context?** <br>
*Alex Sun, Jiayang Wang, Ishan Darji, Ryan Yang* <br>

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://huggingface.co/llejj/Medium_Standard_Transformer/tree/main) from Huggingface.

    Available pretrained models:
    - **Medium Standard Transformer**: `models/linear_regression/8db61f76-d342-4f02-9b37-a7009638c843`
    - **Medium FlashAttention**: `models/linear_regression_flashattn/211fe5c5-7169-4ed1-b6c9-e9e007abaaee`
    - **Medium Mamba**: `models/linear_regression_mamba/981bdb7c-4f7c-4cd0-ae16-8c673d81be5f`
    - **Small Standard Transformer**: `models/linear_regression/6b7b1f9a-bff3-4e71-aa2a-4b08eefdfe24`
    - **Small FlashAttention**: `models/linear_regression/b6e786b1-4452-4a7e-a09b-be179ee5b840`
    - **Small Mamba**: `models/linear_regression/563c7bad-2b16-4015-9af3-551d7382eafe`


3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.
