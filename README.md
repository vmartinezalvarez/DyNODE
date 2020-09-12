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
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | M_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | M_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | M_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | M_LOSS: 0.0000
```
