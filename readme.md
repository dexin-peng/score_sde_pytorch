# Score-Based Diffusion Models with SDE

This repository is a rewrite of [Yang Song's score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)

## Key Features

- Improved dependency management
- Compatible with PyTorch 2.6.0+CUDA 12.6
- Object-oriented style coding
- FID score of 2.388 (nearly the same as the [paper](https://arxiv.org/abs/2011.13456)) on CIFAR-10 using NSCN++ with continuous VESDE
- Docker support for easy deployment and reproducibility
- Easy to extend to other datasets, neural nets, SDEs
- Fixed the wrong prior sampling of the reverse SDE for CIFAR-10, continuous VESDE

## Drawbacks

- Currently only rewrite unconditional CIFAR-10 dataset with NCSN++ architecture and continuous VESDE
- Checkpoints and stats are not compatible with the Checkpoints and stats in [Yang Song's score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- Use Inception-v3 (not v1) to calculate FID
- No Likelihood computation

## Getting Started

### Method 1: Clone and Run

1. Clone this repository:
   ```bash
   git clone https://github.com/dexin-peng/score_sde_pytorch.git
   cd score_sde_pytorch
   ```

2. Start training:
   ```bash
   python3 main.py --config cifar10_nscnpp_cont --mode train
   ```

3. Or generate samples:
   ```bash
   git lfs pull
   # Pull large files, one is the pre-trained checkpoint `assets/ve/cifar10_nscnpp_cont/ckpt/epoch_1999.pth`
   # Another is the stats of CIFAR-10 `data/CIFAR10.npz`
   python3 main.py --config cifar10_nscnpp_cont --mode sample
   ```

### Method 2: Using Docker

A Dockerfile is provided for seperated system and CUDA management:

1. Build the Docker image:
   ```bash
   git clone https://github.com/dexin-peng/score_sde_pytorch.git
   cd score_sde_pytorch
   docker build -t score_sde_pytorch .
   ```

2. Run the container:
   ```bash
   docker run --gpus all -it -p 2222:22 -v $(pwd):/score_sde_pytorch -d score_sde_pytorch
   ```

3. Connect to the container:
   ```bash
   # SSH into the container (replace with your SSH key)
   ssh -p 2222 root@localhost
   cd /score_sde_pytorch
   ```

It's recommended to use `docker compose` to manage

Change the command for your use.


## Command Line Parameters

The following command line parameters are available:

`--config`: (Required) Configuration name to use.

- Currently only `cifar10_nscnpp_cont` rewritten

`--mode`: (Required) Either `train` to train the model or `sample` to generate samples.

`--user_logging_level`: (Optional) Set the logging verbosity. Options: `debug`, `info`, `warning`, `error`. Default: `info`.

`--training_from_scratch`: (Optional) Flag to start training from scratch instead of continuing from a checkpoint.

`--sampling_from_epoch`: (Optional) Specify which training epoch to sample from. Default is the latest available epoch.

## Examples

1. To generate samples from the model:
   ```bash
   python3 main.py --config cifar10_nscnpp_cont --mode sample
   ```
   Uses the configuration `cifar10_nscnpp_cont` and latest checkpoint to generate samples.

2. To sample from a specific training epoch:
   ```bash
   python3 main.py --config cifar10_nscnpp_cont --mode sample --sampling_from_epoch 1999
   ```
   Using the model weights from epoch 1999, allowing you to evaluate the model's performance at that specific point in training.

3. To train the model from scratch:
   ```bash
   python3 main.py --config cifar10_nscnpp_cont --mode train --training_from_scratch
   ```

4. Continue training the model:
   ```bash
   python3 main.py --config cifar10_nscnpp_cont --mode train
   ```

5. All settings are at `config` directory


## Sampling Specification

- Use all 60k CIFAR-10 images to train, and calculate FID with all 60k CIFAR-10 images. [Yang Song's score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) spared 10k images for evaluating per step, and calculate FID based on 50k images. Under the 50k samples to 50k true data settings, I can only reach 2.398 in this 2000 training epochs checkpoint. The best FID is 2.346 with 60k samples to 50k true data settings.

- Followed the `corrector-predictor` sequence, instead of `predictor-corrector`

## The Prior Distribution

- [Yang Song's score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) use 0 as the mean of the prior distribution. [Check the original code](https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/sde_lib.py#L238).

- I found `get_data_scaler` and `get_data_inverse_scaler`. But after careful investigation, I believe the assign prior mean to 0 is wrong. The overall mean of the `train_ds` for CIFAR10 shoule be around 0.473, not 0 (with `config.data.centered=False`)

- If set `config.data.centered=True`, the prior is wrong as well. r,g,b channels mean is `(0.4914*2-1, 0.4822*2-1, 0.4465*2-1)` instead of `(0,0,0)`

- But empirically, the 0 mean implementation also achieves FID 2.465 with 50k to 50k settings.

## Computational Power

- Computational Power Supported by HPC of HKUST(GZ)
