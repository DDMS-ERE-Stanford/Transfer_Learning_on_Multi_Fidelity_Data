import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import copy
import time

#################################################################
#################################################################
def load_GPU_torch():
    '''
    Sets torch device to cuda if cuda is avaliable
    '''
    USE_GPU = True
    dtype = torch.float32 # we will be using float throughout this tutorial
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print('using device:', device)
    return device


def loader_test(data,num_test,Nxy,bs,scale):
    '''
    Builds the data loaders for the test data.
    
    Normally, a test-train split can be done with one function, 
    In this work the procedure is split into 2 differnt functions 
    because when there is a large dataset to conserve vRAM ( we 
    may need to split the process of building training data loaders)
    
    Inputs:
        data - data variable (int)
        num_test - number of test data sets (int)
        Nxy - dimension of data ((int,int))
        bs - batch size (int)
        scale - indicator for fidelity scale of data (str)
    
    Outputs:
        test_loader - data loader for the test data
        data - modified data file with the test data marked as such
    '''
    device = load_GPU_torch()
    
    ks_name = 'ks_' + scale
    ss_name = 'ss_' + scale
    ks_name_test = 'ks_' + scale + '_test'
    ss_name_test = 'ss_' + scale + '_test'
    ks_name_train = 'ks_' + scale  + '_train'
    ss_name_train = 'ss_' + scale  + '_train'
    
    n = len(data[ks_name])
    idx = np.array(torch.randperm(n))
    (nx,ny) = Nxy

    data_idx_ks = data[ks_name][idx]
    data_idx_ss = data[ss_name][idx]
    
    data[ks_name_test] = data_idx_ks[0:num_test]
    data[ss_name_test] = data_idx_ss[0:num_test]
    
    data[ks_name_train] = data_idx_ks[num_test:]
    data[ss_name_train] = data_idx_ss[num_test:]

    ks_dat = data[ks_name_test]
    ss_dat = data[ss_name_test]

    ks_list = list(ks_dat)
    ss_list = list(ss_dat)

    ks_torch = torch.stack([torch.Tensor(np.reshape(i,(1,128,128))) for i in ks_list])
    ss_torch = torch.stack([torch.Tensor(np.reshape(i,(16,nx,ny))) for i in ss_list])

    # shuffle data
    n = len(ks_dat)
    idx = torch.randperm(n)

    ks_shuffle = ks_torch[idx]
    ss_shuffle = ss_torch[idx]

    # split into train, validation, test
    tensor_X = ks_shuffle
    tensor_y = ss_shuffle

    X_test = tensor_X[:num_test]
    y_test = tensor_y[:num_test]

    # pack loaders
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=bs)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    
    return test_loader, data


def loader_train(data, scale, num_training, Nxy, bs, order):
    '''
    Builds the data loaders for the training data.
    
    Normally, a test-train split can be done with one function, 
    In this work the procedure is split into 2 differnt functions 
    because when there is a large dataset to conserve vRAM ( we 
    may need to split the process of building training data loaders)
    
    Inputs:
        data - data variable (int)
        num_test - number of test data sets (int)
        Nxy - dimension of data ((int,int))
        bs - batch size (int)
        scale - indicator for fidelity scale of data (str)
    
    Outputs:
        train_loader - data loader with training data
    '''
    device = load_GPU_torch()
    
    # turn into torch tensor of the correct form
    (nx,ny) = Nxy
    data_start = order*num_training
    data_end = (order+1)*num_training
    data_name = scale + '_train'
    
    ks_train = data['ks_'+data_name][data_start:data_end]
    ss_train = data['ss_'+data_name][data_start:data_end]

    ks_list_train = list(ks_train)
    ss_list_train = list(ss_train)

    ks_torch_train = torch.stack([torch.Tensor(np.reshape(i,(1,128,128))) for i in ks_list_train])
    ss_torch_train = torch.stack([torch.Tensor(np.reshape(i,(16,nx,ny))) for i in ss_list_train])

    # shuffle data
    n_train = len(ks_train)
    idx_train = torch.randperm(n_train)

    X_train = ks_torch_train[idx_train]
    y_train = ss_torch_train[idx_train]

    print(y_train.shape)

    # pack loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bs,shuffle=True)

    return train_loader


def test(epoch, model, test_loader):
    '''
    Evaluates test error at each epoch
    
    Inputs:
        epoch - current epoch (int)
        model - current model being trined 
        test_loader - test loader
    
    Outputs:
        rmse_test - RMSE(Root Mean Square Error) of test
        mae_test - MAE(Mean Absolute Error) of test
    '''
    device = load_GPU_torch()
    
    n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()
    model.eval()
    loss = 0.
    loss_l1 = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            output = model(input)
        loss += F.mse_loss(output, target,size_average=False).item()
        loss_l1 += F.l1_loss(output, target,size_average=False).item()

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    mae_test = loss_l1 / n_out_pixels_test
    # r2_score = 1 - loss / y_test_var
    # print('epoch: {}, test rmse_test:  {:.6f}'.format(epoch, rmse_test))
    return rmse_test, mae_test


def model_train(train_loader, test_loader, reps, n_epochs, log_interval, model_orig, lr, wd, factor, min_lr):
    '''
    Trains model for repetitions designated by "reps" and 
    returns the best model and RMSE obtained by best model
    
    Inputs:
        train_loader - train loader
        test_loader - test_ loader
        reps - number of times to repeat training (int)
        n_epochs - number of epochs to train per each rep (int)
        log_interval - interval(epochs) to compute test error
        model_orig - original model
        lr - learning rate
        wd - weight decay
        factor - factor in ReduceLROnPlateau
        min_lr - minimum learning rate in ReduceLROnPlateau
    
    Outputs:
        model_best - model associated with the lowest test error
        rmse_best - RMSE associated with "model_best"
    '''
    device = load_GPU_torch()
    
    tic = time.time()
    rmse_best = 10**6  #just has to be large
    
    n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
    n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()

    for rep in range(reps):
        rmse_train, rmse_test = [], []
        model = copy.deepcopy(model_orig)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=10,
                                    verbose=True, threshold=0.0001, threshold_mode='rel',
                                    cooldown=0, min_lr=min_lr, eps=1e-08)

        for epoch in range(1, n_epochs+1):
            model.train()
            mse = 0.
            for batch_idx, (input, target) in enumerate(train_loader):
                input, target= input.to(device), target.to(device)
                model.zero_grad()
                output = model(input)
                loss = F.l1_loss(output, target,size_average=False)
                loss.backward()
                optimizer.step()
                mse += F.mse_loss(output, target,size_average=False).item()

            rmse = np.sqrt(mse / n_out_pixels_train)
            scheduler.step(rmse)

            if epoch % log_interval == 0:
                rmse_train.append(rmse)
                rmse_t,_ = test(epoch, model=model, test_loader=test_loader)
                rmse_test.append(rmse_t)

        tic2 = time.time()
        print('Done training {} epochs using {} seconds. Test RMSE = {}'
              .format(n_epochs, tic2 - tic, rmse_t))

        if np.mean(rmse_test[-10:]) < rmse_best:
            model_best = copy.deepcopy(model)
            rmse_best = np.mean(rmse_test[-10:])
    return model_best, rmse_best