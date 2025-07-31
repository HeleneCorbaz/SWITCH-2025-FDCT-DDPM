# SWITCH-2025-FDCT-DDPM

Official repository for the Paper "Towards Diagnostic Quality Flat-Panel Detector CT Imaging Using Diffusion Models" accepted at [SWITCH @ MICCAI 2025](https://switchmiccai.github.io/switch/).

To run the code, we recommend recreating the environment using 
[`mamba`](https://github.com/conda-forge/miniforge) using following command:

    mamba env create -f env.yaml

To run the training or the inference, set the parameter in `config.yaml`
and run `python 2d-ddpm-translation.py` for the training or
`python inference_ddpm.py` for the inference.
