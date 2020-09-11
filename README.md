# DyNODE: Neural Ordinary Differential Equations for Dynamics Modeling in Continuous Control

This repository contains the implementation of [DyNODE: Neural Ordinary Differential Equations for Dynamics Modeling in Continuous Control](https://arxiv.org/pdf/2009.04278.pdf) by Victor M. Martinez Alvarez, Rareș Roșca and Cristian G. Fălcuțescu. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats.

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
