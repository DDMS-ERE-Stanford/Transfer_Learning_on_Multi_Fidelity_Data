# Transfer Learning on Multi-Fidelity Data

Repository for python code used in [Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856)

## Prerequisites

The requirements.txt lists all Python dependencies for this project. To install:
```
  conda install -r requirements.txt
```

## Description of included items
- TLMF_2levels/ML_64_128_transfer.py - script to run experiment to train surrogate model using 64x64 and 128x128 data
- TLMF_2levels/dense_ed_tlmf.py - model construction
- TLMF_2levels/utils_tlmf.py - supporting utility functions for project

## Why use transfer learning on multi-fidelity data for surrogate models?

Neural networks (NNs) are often used as surrogates or emulators of partial differential equations
(PDEs) that describe the dynamics of complex systems. A virtually negligible computational cost of
such surrogates renders them an attractive tool for ensemble-based computation, which requires a
large number of repeated PDE solves. Since the latter are also needed to generate sufficient data for
NN training, the usefulness of NN-based surrogates hinges on the balance between the training cost
and the computational gain stemming from their deployment. Multi-fidelity simulations can be used to 
reduce the cost of data generation for training of a deep convolutional NN(CNN) using transfer learning.

## Numerical Example: Multi Phase Flow
The content of [Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856) 
presents multi-phase flow as the computational example. 

Numerical solution of problems involving multi-phase flow in porous media is notoriously difficult because of the high degree of nonlinearity and stiffness of the governing PDEs. The forward solves of these PDEs using simulation based workflows are computationally expensive.

In the numerical example we consider horizontal flow of two incompressible and immiscible fluids, with viscosities <img src="https://render.githubusercontent.com/render/math?math=\mu_{1}">$\mu_1$ and $\mu_2$, in a heterogeneous, incompressible, and isotropic porous medium $D$. The latter is characterized by porosity $\phi$ and intrinsic permeability $k$. The porosity is assumed to be constant $\phi=0.25$, and intrinsic permeability $k(\mathbf x)$ is treated as a random variable. {\color{blue} The time domain $t$ is between zero and a specified terminal time $T$.} Mass conservation of the $\ell$th fluid phase ($\ell=1,2$) implies

<img src="https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data/blob/745d8e34f872a60f25bc7cb01505a0a90cc04a2c/images/img_log_perm_v4.png" width='500'>

## Acknowlegement
[Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856)

[Convolutional Dense Encoder-Decoder Networks](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)

[Deep Autoregressive Neural Networks for High-Dimensional Inverse Problems in Groundwater Contaminant Source Identification](https://github.com/cics-nd/cnn-inversion)

Last updated 9/30/2021
