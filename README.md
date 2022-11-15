# Learning-ODIN

This repo is the official implementation of our BMVC 2022 paper: Learning ODIN.
# Table of contents
1. [Setup](#setup)
2. [Training with Gradient Quotient](#training)
3. [OOD Detection performance Evaluation](#evaluation)
4. [Pretrained Networks download](#pretrained)

# Paper details
![Learning-ODIN-poster](Learning-ODIN-Poster.png)


## Setup  <a name="setup"></a>

1. We use `conda` as a python package manager. Create an environment using 
   the supplied `environment.yml` file.
2. We follow [MOOD](https://github.com/deeplearning-wisc/MOOD) 
   and use the same datasets for Out-of Distribution detection performance 
   evaluation. We download the datasets form their link ([here](https://drive.google.com/drive/folders/1IRHsD-JRuJP8jUGt0qfFI19-545b3vTd)) and place all datasets under `data` folder. You may create this folder 
   somewhere else and link it to `./data`.

## Training with Gradient Quotient  <a name="training"></a>
We train models with the `main.py` script. For example:
```bash
 python main.py --epochs 200 --name ID-CIFAR-10-MadrysResnet-Gradient-Quotient --which-robust-optimization ODIN-Optimization --lambda-odin=1e-6 --which-odin-reg grad-over-grad --in-dataset CIFAR-10 --id-num-classes 10 --network-name MadrysResnet
```
See `train_all_networks.sh` for the training configuration of all networks 
and In-Distribution datasets.

## Evaluation  <a name="evaluation"></a>
We evaluate OOD detection performance with a [customized version of the 
MOOD](https://github.com/ajevnisek/MOOD) repository.
We copy the trained network from the checkpoints directory to the MOOD 
repository, and run:
```bash
python main.py -ms odin --id cifar10 -mc jpg --arch MadrysResnet --file final_models/ID-CIFAR-10-MadrysResnet-GQ.pth
```
A best practice approach is to pipe the script output into a file:
```bash
mkdir logs
python main.py -ms odin --id cifar10 -mc jpg --arch MadrysResnet --file final_models/ID-CIFAR-10-MadrysResnet-GQ.pth > logs/ID-CIFAR-10-MadrysResnet-GQ.log
```
## Pretrained networks with GQ:  <a name="pretrained"></a>
You can our pretrained models from [here](https://drive.google.com/drive/folders/1PsuRRuVcbf0lqNJwzhQBTaJAgoSoa0n9?usp=share_link).

