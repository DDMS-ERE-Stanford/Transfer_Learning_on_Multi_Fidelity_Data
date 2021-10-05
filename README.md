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

## Methodology
In this project, the CNN is trained using high-fidelity-simulations(HFS) and low-fidelity-simulations(LFS). A detailed description of the methodology and algorithm are provided in [Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856). An abridged methodology is provided below:

<img src="https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data/blob/4272638e0e038898f4ed3239abf0a002b359a5aa/images/img_method_v9.png" width='1200'>

- Phase1: Train CNN layers LFS
  - State1.1: Initialize encoder-decoder model (M)
  - State1.2: The last layer of the M is replaced with a temporary layer. This modified model (M1) has outputs which match the dimensions of LFS. M1 is trained on LFS
- Phase2: Train original last layer using HFS
  - State2.1: Replace temporary layer in M1 with the original last layer from M. This mododified model (M2) now has the output dimensions of HFS. 
  - State2.2: Lock the all of the weights trained in Phase1 so that last layer of M2 can be updated. Train M2 using HFS.
 - Phase3: Fine-tune entire CNN by unlocking all weights and training on HFS

A snapshot of multi-scale data is provided below:

<img src="https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data/blob/4272638e0e038898f4ed3239abf0a002b359a5aa/images/img_multi_scale_v5.png" width='650'>

## Numerical Example: Multi Phase Flow
Numerical solution of problems involving multi-phase flow in porous media is notoriously difficult because of the high degree of nonlinearity and stiffness of the governing PDEs. The forward solves of these PDEs using simulation based workflows are computationally expensive. The content of [Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856) presents multi-phase flow as the computational example and includes a detailed problem description.

The input to the CNN surrogate model are random permeability field and the output are snapshots of the saturation maps at 16 different time steps. The dimensions of the input should be (128x128) and the dimensions of the output should be (16x128x128).

An example of the log permeability field is provided below:

<img src="https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data/blob/745d8e34f872a60f25bc7cb01505a0a90cc04a2c/images/img_log_perm_v4.png" width='400'>

A comparison of the data(left columns), model(middle columns), and the difference(right columns) from the test data set are provided below:

<img src="https://github.com/DDMS-ERE-Stanford/Transfer_Learning_on_Multi_Fidelity_Data/blob/4272638e0e038898f4ed3239abf0a002b359a5aa/images/img_results_diff_v8.png" width='1200'>


## Acknowledgement
[Transfer Learning on Multi-Fidelity Data](https://arxiv.org/abs/2105.00856)

[Convolutional Dense Encoder-Decoder Networks](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)

[Deep Autoregressive Neural Networks for High-Dimensional Inverse Problems in Groundwater Contaminant Source Identification](https://github.com/cics-nd/cnn-inversion)

Last updated 9/30/2021
