# Transfer Learning on Multi-Fidelity Data

## Prerequisites

The requirements.txt lists all Python dependencies for this project. To install:
```
  conda install -r requirements.txt
```

## Description of included items
- TLMF_2levels/ML_64_128_transfer.py - experiment to train surrogate model using 64x64 and 128x128 data
- TLMF_2levels/dense_ed_tlmf.py -  model construction
- TLMF_2levels/utils_tlmf.py - supporting utility functions for project

## Why use transfer learning on multi-fidelity data for surrogate models?

Neural networks (NNs) are often used as surrogates or emulators of partial differential equations
(PDEs) that describe the dynamics of complex systems. A virtually negligible computational cost of
such surrogates renders them an attractive tool for ensemble-based computation, which requires a
large number of repeated PDE solves. Since the latter are also needed to generate sufficient data for
NN training, the usefulness of NN-based surrogates hinges on the balance between the training cost
and the computational gain stemming from their deployment. Multi-fidelity simulations can be used to 
reduce the cost of data generation for training of a deep convolutional NN(CNN) using transfer learning.

## Acknowlegement
[Convolutional Dense Encoder-Decoder Networks](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)

Last updated 9/30/2021
