# DyNODE: Neural Ordinary Differential Equations for Dynamics Modeling in Continuous Control

This repository contains the implementation of [DyNODE: Neural Ordinary Differential Equations for Dynamics Modeling in Continuous Control](https://arxiv.org/pdf/2009.04278.pdf) by Victor M. Martinez Alvarez, Rareș Roșca and Cristian G. Fălcuțescu. Our implementation of SAC is based on [CURL](https://github.com/MishaLaskin/curl) by Misha Laskin.

Experiments are done in PyTorch. If you find this repository helpful, please cite our work:

```
@article{alvarez2020dynode,
	author    = {Victor M. Martinez Alvarez and Rareș Roșca and Cristian G. Fălcuțescu},
	title     = {DyNODE: Neural Ordinary Differential Equations for Dynamics Modeling in Continuous Control},
	journal   = {arXiv:2009.04278},
	year      = {2020},
}
```

## Installation 

All the required dependencies are in the `conda_env.yml` file. To install them use the following command:

```
conda env create -f conda_env.yml
```

## Instructions
To train a DyNODE agent on the `cartpole swingup` task run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.
```
CUDA_VISIBLE_DEVICES=0 python train_dynode.py \
    --domain_name cartpole \
    --task_name swingup \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 5000 --batch_size 128 --num_train_steps 100001 
```

In your console, you should see printouts that look like:


```
| eval  | S: 5000 | ER: 167.09909
| train | E: 0 | S: 5000 | D: 301.1 s | R: 0.00000   | BR: 0.00000 | A_LOSS: 0.00000   | CR_LOSS: 0.00000 | M_LOSS: 0.00000
| train | E: 6 | S: 6000 | D: 281.9 s | R: 186.28232 | BR: 0.13017 | A_LOSS: -18.17264 | CR_LOSS: 0.18433 | M_LOSS: 0.08837
| train | E: 7 | S: 7000 | D: 275.6 s | R: 163.70892 | BR: 0.13792 | A_LOSS: -20.57893 | CR_LOSS: 0.47372 | M_LOSS: 0.03110
| train | E: 8 | S: 8000 | D: 274.1 s | R: 242.66144 | BR: 0.15333 | A_LOSS: -22.16228 | CR_LOSS: 0.54115 | M_LOSS: 0.07415
| train | E: 9 | S: 9000 | D: 271.4 s | R: 207.88210 | BR: 0.17214 | A_LOSS: -23.22978 | CR_LOSS: 0.31086 | M_LOSS: 0.04055
| train | E: 10 | S: 10000 | D: 0.0 s | R: 468.76406 | BR: 0.19523 | A_LOSS: -24.92557 | CR_LOSS: 0.24657 | M_LOSS: 0.01961
| eval  | S: 10000 | ER: 478.78332

```
