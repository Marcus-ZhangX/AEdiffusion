# Time : 2023/7/4 12:38
# Tong ji Marcus
# FileName: run_surrogate_zhengna.py
import argparse
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
import torch as th
import numpy as np
import sys
import os
import pprint
from torchinfo import summary
import h5py
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import torch.autograd as autograd
from torch import optim
import pandas as pd
import scipy.io

device = th.device("cuda" if th.cuda.is_available() else "cpu")
# 决定了几个卷积层
class _DenseLayer(nn.Sequential):

    def __init__(self, in_features, growth_rate, drop_rate=0, bn_size=4, bottleneck=False):
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

            self.add_module('conv1',
                            nn.Conv2d(in_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            y = F.dropout2d(y, p=self.drop_rate, training=self.training)  # dropout函数防止过拟合（暂时丢弃部分节点
        z = torch.cat([x, y], 1)
        return z


# 一个稠密块中包含多少层上面定义的DenseLayer
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate, bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate, drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)


# encoder和decoder的转换
class _Transition(nn.Sequential):

    def __init__(self, in_features, out_features, encoding=True, drop_rate=0., last=False, out_channels=3,
                 outsize_even=True):
        super(_Transition, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))

        # encoding设置
        if encoding:
            # reduce feature maps; half image size (input feature size is even)
            # bottleneck impl, save memory, add nonlinearity
            self.add_module('conv1',
                            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False))

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
            # decoder设置
            if last:
                ks = 6 if outsize_even else 3
                out_convt = nn.ConvTranspose2d(out_features, out_channels, kernel_size=ks, stride=2, padding=1,
                                               bias=False)
            else:
                out_convt = nn.ConvTranspose2d(out_features, out_features, kernel_size=3, stride=2, padding=1,
                                               output_padding=0, bias=False)

            # bottleneck impl, save memory, add nonlinearity
            self.add_module('conv1',
                            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=False))

            if drop_rate > 0:
                self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

            self.add_module('norm2', nn.BatchNorm2d(out_features))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('convT2', out_convt)
            if drop_rate > 0:
                self.add_module('dropout2', nn.Dropout2d(p=drop_rate))


class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, growth_rate=16,
                 num_init_features=64, bn_size=4, drop_rate=0, outsize_even=False,
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

        if len(blocks) > 1 and len(blocks) % 2 == 0:
            ValueError('length of blocks must be an odd number, but got {}'.format(len(blocks)))

        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]
        self.features = nn.Sequential()
        self.features.add_module('in_conv',
                                 nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                           bias=False))
        # self.features.add_module('in_conv',nn.Conv2d(in_channels, num_init_features,kernel_size=6, stride=2, padding=2, bias=False))

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

        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, bottleneck=bottleneck)
            self.features.add_module('decblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate

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
        y[:, 0] = F.softplus(y[:, 0].clone(), beta=1)

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


#---------------------------------------------------------------------------------

# 参数设置
parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='single', help='experiment name')
parser.add_argument('--skip', action='store_true', default=False, help='enable skip connection between encoder and decoder nets')
parser.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')
parser.add_argument('--data-dir', type=str, default="/afs/crc.nd.edu/user/s/smo/invers_mt3d/DCEDN_sequence_2out/", help='data directory')
parser.add_argument('--kle-terms', type=int, default=679, help='num of KLE terms')
parser.add_argument('--n-train', type=int, default=3000, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")
parser.add_argument('--loss-fn', type=str, default='l1', help='loss function: mse, l1, huber, berhu')
parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.002, help='learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 100)')
parser.add_argument('--log-interval', type=int, default=5, help='how many epochs to wait before logging training status')
parser.add_argument('--plot-interval', type=int, default=50, help='how many epochs to wait before plotting training status')

args = parser.parse_args(args=[])

# 实例化模型
model = DenseED(3, 2, blocks=args.blocks, growth_rate=args.growth_rate,
                        drop_rate=args.drop_rate, bn_size=args.bn_size,
                        num_init_features=args.init_features, bottleneck=args.bottleneck).to(device)


# 导入模型
model_dir = "./"
model.load_state_dict(torch.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
print('Loaded model')