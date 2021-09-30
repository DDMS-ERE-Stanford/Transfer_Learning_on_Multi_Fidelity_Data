import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Transfer Learning on Multi-Fidelity Data

Reference:
    https://github.com/DDMS-ERE-Stanford/Multi_Level_Surrogate_Model

This work adds functions to manupulate the model found in:
    "Convolutional Dense Encoder-Decoder Networks":
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    
Dong Hee Song
Mar 31,2021
Sep 29,2021
"""


class _DenseLayer(nn.Sequential):  #From "Convolutional Dense Encoder-Decoder Networks"
    # bottleneck layer, bn_size: bottleneck size
    def __init__(self, in_features, growth_rate, drop_rate=0, bn_size=4,
                 bottleneck=False):
        # detect if the input features are more than bn_size x k,
        # if yes, use bottleneck -- not much memory gain, but lose one relu
        # I disabled the bottleneck for current implementation
        super(_DenseLayer, self).__init__()
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(in_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            y = F.dropout2d(y, p=self.drop_rate, training=self.training)
        z = torch.cat([x, y], 1)
        return z


class _DenseBlock(nn.Sequential):  #From "Convolutional Dense Encoder-Decoder Networks"
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):  #From "Convolutional Dense Encoder-Decoder Networks"
    def __init__(self, in_features, out_features, encoding=True, drop_rate=0.,
                 last=False, out_channels=3, outsize_even=True):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if encoding:
            # reduce feature maps; half image size (input feature size is even)
            # bottleneck impl, save memory, add nonlinearity
            self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                              kernel_size=1, stride=1,
                                              padding=0, bias=False))
            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                              kernel_size=3, stride=2,
                                              padding=1, bias=False))
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
        else:
            # decoding, transition up
            if last:
                ks = 4  # if outsize_even else 5
                out_convt = nn.ConvTranspose2d(out_features, out_channels,
                                kernel_size=ks, stride=2, padding=1, bias=False)
            else:
                out_convt = nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=4, stride=2,
                    padding=1, output_padding=0, bias=False)

            # bottleneck impl, save memory, add nonlinearity
            self.add_module('conv1', nn.Conv2d(in_features, out_features,
                                              kernel_size=1, stride=1,
                                              padding=0, bias=False))
            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('convT2', out_convt)
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout2d(p=drop_rate))

                
class DenseED(nn.Module):  #From "Convolutional Dense Encoder-Decoder Networks"
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 num_init_features=64, bn_size=4, drop_rate=0, outsize_even=True,
                 bottleneck=False):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            blocks: list (of odd size) of integers
            growth_rate (int): K
            num_init_features (int): the number of feature maps after the first
                conv layer
            bn_size: bottleneck size for number of feature maps (not useful...)
            bottleneck (bool): use bottleneck for dense block or not
            drop_rate (float): dropout rate
            outsize_even (bool): if the output size is even or odd (e.g.
                65 x 65 is odd, 64 x 64 is even)

        """
        super(DenseED, self).__init__()
        self.out_channels = out_channels

        if len(blocks) > 1 and len(blocks) % 2 == 0:
            ValueError('length of blocks must be an odd number, but got {}'
                       .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]
        self.features = nn.Sequential()
        # First convolution ================
        # only conv, half image size
        self.features.add_module('in_conv',
                    nn.Conv2d(in_channels, num_init_features,
                            kernel_size=5, stride=2, padding=2, bias=False))

        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = num_init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, bottleneck=bottleneck)
            self.features.add_module('encblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans = _Transition(in_features=num_features,
                                out_features=num_features // 2,
                                encoding=True, drop_rate=drop_rate)
            self.features.add_module('down%d' % (i + 1), trans)
            num_features = num_features // 2

        # Decoding / transition up ==============
        # dense block --> decoding --> dense block --> decoding --> dense block
        # if len(dec_block_layers) - len(enc_block_layers) == 1:
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, bottleneck=bottleneck)
            self.features.add_module('decblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate

            # if this is the last decoding layer is the output layer
            last_layer = True if i == len(dec_block_layers) - 1 else False

            trans = _Transition(in_features=num_features,
                                out_features=num_features // 2,
                                encoding=False, drop_rate=drop_rate,
                                last=last_layer, out_channels=out_channels,
                                outsize_even=outsize_even)
            self.features.add_module('up%d' % (i + 1), trans)
            num_features = num_features // 2

    def forward(self, x):
        y = self.features(x)
        
        # use the softplus activation for concentration
#         y[:,:self.out_channels-1] = F.softplus(y[:,:self.out_channels-1].clone(), beta=5)
        y = F.softplus(y.clone(), beta=5)
        # in the example, pressure is the last output channel
        # use the sigmoid activation for pressure
#         y[:,self.out_channels-1] = torch.sigmoid(y[:,self.out_channels-1])

        return y

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))

                        
class DenseED_phase1(nn.Module):
    def __init__(self,model,blocks):
        '''
        Modifies the DenseED model to remove the "up2" layer and
        adds a temporary layer to force output to match
        dimensions of IFS [64x64]
        
        Input:
            model: model as described by DenseED
            blocks: list (of odd size) of integers describing DenseED
        '''
        super(DenseED_phase1, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('in_conv', model.features.in_conv)
        self.features.add_module('encblock1', model.features.encblock1)
        self.features.add_module('down1', model.features.down1)
        self.features.add_module('decblock1', model.features.decblock1)
        self.features.add_module('up1', model.features.up1)
        self.features.add_module('decblock2', model.features.decblock2)
#         self.features.add_module('up2', model.features.up2)        
        # Adding layer to train on 1x16x64x64
        last_in = model.features.decblock2.denselayer1.conv1.in_channels
        last_out = model.features.decblock2.denselayer1.conv1.out_channels
        self.features.add_module(
            'temp_out',
            nn.Conv2d(
                in_channels=int(last_in+last_out*blocks[2]),
                out_channels=16,
                kernel_size=1, stride=1, padding=0, bias=False
            )
        )

        
    def forward(self, x):
        y = self.features(x)
        y = F.softplus(y.clone(), beta=5)
        return y

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
                        
                        
class DenseED_phase2(nn.Module):
    def __init__(self,model_p0,model_p1):
        '''
        Restores the form of DenseED but uses the layers trained in 
        Phase1 using DenseED_phase1
        
        Input:
            model_p0 - model from original DenseED 
            podel_p1 - model from DenseED_phase1
        '''
        super(DenseED_phase2, self).__init__()
        # from model_p1 (part1)
        self.features = nn.Sequential()
        self.features.add_module('in_conv', model_p1.features.in_conv)
        self.features.add_module('encblock1', model_p1.features.encblock1)
        self.features.add_module('down1', model_p1.features.down1)
        self.features.add_module('decblock1', model_p1.features.decblock1)
        self.features.add_module('up1', model_p1.features.up1)
        self.features.add_module('decblock2', model_p1.features.decblock2)
        
        # from model_p0 (part0, original model)
        self.features.add_module('up2', model_p0.features.up2)

        
    def forward(self, x):
        y = self.features(x)
        y = F.softplus(y.clone(), beta=5)
        return y

    def _num_parameters_convlayers(self):
        n_params, n_conv_layers = 0, 0
        for name, param in self.named_parameters():
            if 'conv' in name:
                n_conv_layers += 1
            n_params += param.numel()
        return n_params, n_conv_layers

    def _count_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            print(name)
            print(param.size())
            print(param.numel())
            n_params += param.numel()
            print('num of parameters so far: {}'.format(n_params))

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            # pass self, otherwise infinite loop
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))
