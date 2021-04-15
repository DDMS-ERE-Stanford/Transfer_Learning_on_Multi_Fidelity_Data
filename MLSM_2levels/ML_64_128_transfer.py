import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.autograd import Variable
from h5py import File
import h5py
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import copy
import time

from dense_ed_ddms import *
from utils_ddms import *

'''
"Accelerated training of neural networks via multi-fidelity simulations"
    https://github.com/DDMS-ERE-Stanford/Multi_Level_Surrogate_Model.git

Sample code to train model on 128x128(hfs) and 64x64(ifs) data. 
Surrogate model will be trained on muitiple fidelity of data.
Final model will be saved.

The original encoder-decoder model was based on that of:
    "Convolutional Dense Encoder-Decoder Networks":
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

Stanford University, DDMS group

Dong Hee Song (dhsong@stanford.edu)
Mar31,2021
'''

tic_all = time.time()


'''
#################################################################
#################################################################
Experiment controls
#################################################################
#################################################################
'''
# Data quantity selection
n_hfs = 100   # number of high (128x128) fidelity data
n_ifs = 300   # number of intermediate (64x64) fidelity data
n_test = 100  # number of test data

model_file_name = "model_2layer.pth"

# PHASE1 Experimental controls
reps_phase1 = 2        # repetitions for phase1
n_epochs_phase1 = 50   # epochs in phase1
lr_phase1=0.0005       # learning rate in phase1
wd_phase1=1e-5         # weight decay in phase1
factor_phase1=0.6      # factor for "ReduceLROnPlateau" in phase1
min_lr_phase1=1.5e-06  # minimum learning rate in phase1

# PHASE2 Experimental controls
reps_phase2 = 2        # repetitions for phase2
n_epochs_phase2 = 50   # epochs in phase2
lr_phase2=0.00005      # learning rate in phase2
wd_phase2=1e-5         # weight decay in phase2
factor_phase2=0.6      # factor for "ReduceLROnPlateau" in phase2
min_lr_phase2=1.5e-06  # minimum learning rate in phase2

# PHASE3 Experimental controls
reps_phase3 = 2        # repetitions for phase3
n_epochs_phase3 = 50   # epochs in phase3
lr_phase3=0.00001      # learning rate in phase3
wd_phase3=1e-5         # weight decay in phase3
factor_phase3=0.6      # factor for "ReduceLROnPlateau" in phase3
min_lr_phase3=5e-07    # minimum learning rate in phase3

'''
#################################################################
#################################################################
Device Selection (CUDA)
#################################################################
#################################################################
'''
USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

'''
#################################################################
#################################################################
Import 128x128 data

Reads the data in the .hdf5 files and adds it to "dat" variable
#################################################################
#################################################################
'''
path = '/data/cees/dhsong/ML_Train/P2_data_files/'

filename = 'DATA_het_128x128_v1.hdf5'
filename_full = path + filename
print(filename_full)
f = File(filename_full,'r')

# the data needs to be transposed as (3,2,0,1) to get images from matlab convention the form we want
ks = np.transpose(f['ks'], (3,2,0,1)) * (10**14)
ps = np.transpose(f['ps'], (3,2,0,1))
ss = np.transpose(f['ss'], (3,2,0,1))
f.close()

dat = {}  # primary data variable
dat['ks_hfs'] = ks
dat['ps_hfs'] = ps
dat['ss_hfs'] = ss
##########################################################
##########################################################
filename = 'DATA_het_128x128_v2.hdf5'
filename_full = path + filename
print(filename_full)
f = File(filename_full,'r')

# the data needs to be transposed as (3,2,0,1) to get images back
ks = np.transpose(f['ks'], (3,2,0,1)) * (10**14)
ps = np.transpose(f['ps'], (3,2,0,1))
ss = np.transpose(f['ss'], (3,2,0,1))
f.close()

dat['ks_hfs'] = np.concatenate((dat['ks_hfs'],ks),axis=0)
dat['ps_hfs'] = np.concatenate((dat['ps_hfs'],ps),axis=0)
dat['ss_hfs'] = np.concatenate((dat['ss_hfs'],ss),axis=0)

print(dat['ks_hfs'].shape)
print(dat['ps_hfs'].shape)
print(dat['ss_hfs'].shape)

'''
#################################################################
#################################################################
Import 64x64 data

Reads the data in the .hdf5 files and adds it to "dat" variable
#################################################################
#################################################################
'''

filename = 'DATA_het_upscale_64x64_v1.hdf5'
filename_full = path + filename
print(filename_full)
f = File(filename_full,'r')

# the data needs to be transposed as (3,2,0,1) to get images back
ks = np.transpose(f['ks_fine'], (3,2,0,1)) * (10**14)
ps = np.transpose(f['ps'], (3,2,0,1))
ss = np.transpose(f['ss'], (3,2,0,1))
f.close()

dat['ks_ifs'] = ks
dat['ps_ifs'] = ps
dat['ss_ifs'] = ss
##########################################################
##########################################################
filename = 'DATA_het_upscale_64x64_v2.hdf5'
filename_full = path + filename
print(filename_full)
f = File(filename_full,'r')

# the data needs to be transposed as (3,2,0,1) to get images back
ks = np.transpose(f['ks_fine'], (3,2,0,1)) * (10**14)
ps = np.transpose(f['ps'], (3,2,0,1))
ss = np.transpose(f['ss'], (3,2,0,1))
f.close()

dat['ks_ifs'] = np.concatenate((dat['ks_ifs'],ks),axis=0)
dat['ps_ifs'] = np.concatenate((dat['ps_ifs'],ps),axis=0)
dat['ss_ifs'] = np.concatenate((dat['ss_ifs'],ss),axis=0)

print(dat['ks_ifs'].shape)
print(dat['ps_ifs'].shape)
print(dat['ss_ifs'].shape)

'''
#################################################################
#################################################################
Phase1 - experiment [IFS]

- Build data-loaders using IFS data
- Build model and modify to build model_phase1
- Train model_phase1 using IFS data
#################################################################
#################################################################
'''
# Build data loaders
test_loader_ifs, dat = \
    loader_test(data=dat,num_test=n_test,Nxy=(64,64),bs=40,scale='ifs')

train_loader = loader_train(data=dat, scale='ifs', \
                            num_training=n_ifs, Nxy=(64,64), bs=40, order=0)

#########################################################
#########################################################
# Build model
model_orig = None
model_orig = DenseED(1, 16, blocks=(7,12,7), growth_rate=40,
                        drop_rate=0, bn_size=8,
                        num_init_features=64, bottleneck=False).to(device)
model_phase1_orig = DenseED_phase1(model_orig,blocks=(7,12,7)).to(device)

##########################################################
##########################################################

model_phase1, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_ifs, 
                                      reps=reps_phase1, n_epochs=n_epochs_phase1, log_interval=1, 
                                      model_orig=model_phase1_orig, 
                                      lr=lr_phase1, wd=wd_phase1, factor=factor_phase1, min_lr=min_lr_phase1)
print(f"PHASE1 RMSE BEST = {rmse_best}")
    
'''
#################################################################
#################################################################
Phase2 - experiment [HFS1]

- Build data-loaders using HFS data
- Build modify model_phase1 to build model_phase2
- Train model_phase2 using HFS data
#################################################################
#################################################################
'''
# Build data loaders
test_loader_hfs, dat = \
    loader_test(data=dat,num_test=n_test,Nxy=(128,128),bs=40,scale='hfs')

train_loader = loader_train(data=dat, scale='hfs', \
                                      num_training=n_hfs, Nxy=(128,128), bs=40, order=0)
#########################################################
#########################################################
# Build model
model_phase2_orig = DenseED_phase2(model_orig,model_phase1)

# Freeze weights from being updated (whole model)
for param in model_phase2_orig.parameters():
    param.requires_grad = False

# Unfreeze the parts we do want to update
for param in model_phase2_orig.features.decblock2.parameters():
    param.requires_grad = True
for param in model_phase2_orig.features.up2.parameters():
    param.requires_grad = True

#########################################################
#########################################################
# Train
model_phase2, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_hfs, 
                                      reps=reps_phase2, n_epochs=n_epochs_phase2, log_interval=1, 
                                      model_orig=model_phase2_orig, 
                                      lr=lr_phase2, wd=wd_phase2, factor=factor_phase2, min_lr=min_lr_phase2)
print(f"PHASE2 RMSE BEST = {rmse_best}")

'''
#################################################################
#################################################################
Phase3 - experiment [HFS2]

- Build modify model_phase2 to build model_phase3
- Train model_phase3 using HFS data
#################################################################
#################################################################
'''
# Build model
model_phase3_orig = model_phase2
# Unfreeze all weights
for param in model_phase3_orig.parameters():
    param.requires_grad = True
#########################################################
#########################################################
# Train
model_phase3, rmse_best = model_train(train_loader=train_loader, test_loader=test_loader_hfs, 
                                      reps=reps_phase3, n_epochs=n_epochs_phase3, log_interval=1, 
                                      model_orig=model_phase3_orig, 
                                      lr=lr_phase3, wd=wd_phase3, factor=factor_phase3, min_lr=min_lr_phase3)
print(f"PHASE3 RMSE BEST = {rmse_best}")

# Save model
torch.save(model_phase3.state_dict(), model_file_name)

toc_all = time.time()
print('EXPERIMENT COMPLETE, TIME ELAPSED: {} [sec]'
      .format(toc_all-tic_all))
