from setuptools import setup, find_packages

# Note: This package is still under the context_general_bci namespace, same as the codebase it inherited (https://github.com/joel99/context_general_bci)
# Thus installing both packages will lead to namespace conflicts.
# Presuming this is fine b/c NDT3 essentially deprecates NDT2 in terms of results.

# Switching is currently onerous because preprocessed data is locked to the `context_general_bci` namespace.
setup(
    name='context_general_bci',
    version='0.1.0',

    url='https://github.com/joel99/ndt3',
    author='Joel Ye',
    author_email='joelye9@gmail.com',
    description='NDT3 pretraining and analysis monorepo',
    packages=find_packages(exclude=['scripts', 'crc_scripts', 'data', 'data_scripts', 'slurm']),
    py_modules=['context_general_bci'],

    install_requires=[
        'torch==2.1.0+cu118', # 2.0 onnx export doesn't work, install with --extra-index-url https://download.pytorch.org/whl/cu117
        'torchvision==0.16.0+cu118', # For flash-attn compat
        'seaborn',
        'pandas',
        'numpy',
        'scipy',
        # 'onnxruntime-gpu',
        'pyrtma',
        'hydra-core',
        'yacs',
        'pynwb',
        'argparse',
        'wandb',
        'einops',
        'lightning',
        'scikit-learn',
        'ordered-enum',
        'mat73',
        'dacite',
        'gdown',
        'timm',
        'pyrtma', # For realtime Pitt infra
        'transformers', # Flash Attn
        'peft',
        'packaging', # Flash Attn https://github.com/Dao-AILab/flash-attention
        'ninja',
        'rotary-embedding-torch', # https://github.com/lucidrains/rotary-embedding-torch
        'sentencepiece', # Flash Attn
        'edit-distance',
        'falcon-challenge',
        'ruamel.yaml',
        'tensordict>=0.3.0',
        # 'flash-attn', # install following build instructions on https://github.com/Dao-AILab/flash-attention
        # Add nvcc corresponding to torch (module system on cluster, cuda/11.8)
        # -- export CUDA_HOME=/ihome/crc/install/cuda/11.8
        # pip install flash-attn --no-build-isolation
    ],
)