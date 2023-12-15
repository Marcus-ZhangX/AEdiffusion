# Time : 2023/9/10 0:58
# Tong ji Marcus
# FileName: ARnet_assessment.py

import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
                 num_init_features=64, bn_size=4, drop_rate=0, outsize_even=True,
                 bottleneck=False):
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



# default to use cuda
parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='AR-Net', help='experiment name')
parser.add_argument('--blocks', type=list, default=(5, 10, 5),
                    help='list of number of layers in each block in decoding net')  # 解码网络中每一块的层数列表
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')  # 每层卷积的输出，不改
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--init-features', type=int, default=48,
                    help='# initial features after the first conv layer')  # 初始卷积层输出的特征面数量

parser.add_argument('--data-dir', type=str, default="E:/data_and_code/data/data_for_ARnet/Surrogate_model/", help='data directory')

parser.add_argument('--n-train', type=int, default=3000, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")

parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0002, help='learnign rate')

parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 100)')
parser.add_argument('--log-interval', type=int, default=1,
                    help='how many epochs to wait before logging training status')
parser.add_argument('--plot-interval', type=int, default=2,
                    help='how many epochs to wait before plotting training status')
args = parser.parse_args(args=[])
device = th.device("cuda" if th.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

model = DenseED(3, 2, blocks=args.blocks, growth_rate=args.growth_rate,
                drop_rate=args.drop_rate, bn_size=args.bn_size,
                num_init_features=args.init_features, bottleneck=args.bottleneck).cuda()

model_dir = r"E:\Coding_path\DiffuseVAE\scripts"
model.load_state_dict(torch.load(model_dir + '/model_epoch{}.pth'.format(200)))
print('Loaded model')

#----------------------------------------------------------------------------------------------------
# 画出ARnet的输出的图
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 读取hdf5文件
# with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/input_lhs3000_train.hdf5".format(args.n_train), 'r') as f:
#     x_train = f['dataset'][()]
#     print("train input data shape: {}".format(x_train.shape))
# with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/output_lhs3000_train.hdf5".format(args.n_train), 'r') as f:
#     y_train = f['dataset'][()]
#     print("train output data shape: {}".format(y_train.shape))

with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/input_lhs500_testing.hdf5".format(args.n_test), 'r') as f:
    x_test = f['dataset'][()]
    print("test input data shape: {}".format(x_test.shape))
with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/output_lhs500_testing.hdf5".format(args.n_test), 'r') as f:
    y_test = f['dataset'][()]
    print("test output data shape: {}".format(y_test.shape))


ntimes = 5
n_samples = 1
idx = th.LongTensor(np.random.choice(500, n_samples, replace=False))  # 随机取test数据中的1个
np.seterr(divide='ignore', invalid='ignore')
# idx = th.LongTensor([13])
print("Index of data: {}".format(idx))
print("X shape: {}".format(x_test.shape))

for i in range(n_samples):
    model.eval()
    x = x_test[idx[i] * ntimes: (idx[i] + 1) * ntimes]
    y = y_test[idx[i] * ntimes: (idx[i] + 1) * ntimes]

    y_output = np.full((ntimes, y_test.shape[1], 128, 128), 0.0)  # 1
    x_ii = np.full((1, x_test.shape[1], 128, 128), 0.0)
    y_ii_1 = np.full((128, 128), 0.0)  # y_0 = 0
    for ii in range(ntimes):
        x_ii[0, 0, :, :] = x[ii, 0, :, :]  # hydraulic conductivity
        x_ii[0, 1, :, :] = x[ii, 1, :, :]  # source rate
        x_ii[0, 2, :, :] = y_ii_1  # the ii_th predicted output
        x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
        with th.no_grad():
            y_hat = model(x_ii_tensor)
        y_hat = y_hat.data.cpu().numpy()
        y_output[ii] = y_hat
        y_ii_1 = y_hat[0, 0, :, :]  # treat the current output as input to predict the ouput at next time step

    y_target = np.full((ntimes + 1, 1, 128, 128), 0.0)
    y_target[:ntimes] = y[:, [0]]  # the concentration fields for one input conductivity fields at ntimes time steps
    y_target[ntimes] = y[[0], [1]]  # 水头

    y_pred = np.full((ntimes + 1, 1, 128, 128), 0.0)
    y_pred[:ntimes] = y_output[:, [0]]
    y_pred[ntimes] = y_output[[0], [1]]  # 预测水头

    y_relative = (y_target - y_pred) / y_target
    samples = np.vstack(y_target, y_pred, y_relative)  # np.vstack保持列不变进行拼接
    # print(samples.shape)

    small_tensors = np.split(samples, 18, axis=0)
    save_path = r'E:\Coding_path\DiffuseVAE\scripts\ARnet_prediction'
    # 保存参考场的colorbar的上下限
    a = np.zeros((5, 2))
    for i, small_tensor in enumerate(small_tensors):
        if i > -1 and i < 5:  # 0到4，五张图
            small_tensor = small_tensor * 1000
            scaled_tensor = np.log10(small_tensor + 1)
            vmin, vmax = np.min(scaled_tensor), np.max(scaled_tensor)
            # print(vmin)
            # print(vmax)
            a[i, 0] = vmin
            a[i, 1] = vmax

    for i, small_tensor in enumerate(small_tensors):
        if i > -1 and  i < 5:  #0到4，五张图
            small_tensor = small_tensor*1000
            scaled_tensor = np.log10(small_tensor+1)

            vmin, vmax = np.min(scaled_tensor), np.max(scaled_tensor)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(scaled_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            ticks = [vmin,vmin+(vmax-vmin)/10,vmin+(vmax-vmin)/10*2,vmin+(vmax-vmin)/10*3,vmin+(vmax-vmin)/10*4,
                        vmin+(vmax-vmin)/10*5,vmin+(vmax-vmin)/10*6,vmin+(vmax-vmin)/10*7,vmin+(vmax-vmin)/10*8
                     ,vmin+(vmax-vmin)/10*9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()
        elif i== 5:
            vmin, vmax = np.min(small_tensor), np.max(small_tensor)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(small_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
                     vmin + (vmax - vmin) / 10 * 4,
                     vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6, vmin + (vmax - vmin) / 10 * 7,
                     vmin + (vmax - vmin) / 10 * 8
                , vmin + (vmax - vmin) / 10 * 9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()
        elif i > 5 and i < 11:
            small_tensor = small_tensor * 1000
            scaled_tensor = np.log10(small_tensor+1)
            vmin, vmax = np.min(scaled_tensor), np.max(scaled_tensor)
            # vmin = a[i-6, 0]
            # vmax = a[i-6, 1]
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(scaled_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
                     vmin + (vmax - vmin) / 10 * 4,vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6, vmin + (vmax - vmin) / 10 * 7,
                     vmin + (vmax - vmin) / 10 * 8
                , vmin + (vmax - vmin) / 10 * 9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()
        elif i == 11:
            # scaled_tensor =10 + (small_tensor - vmin1) / (vmax1 - vmin1)
            small_tensor[small_tensor > 11] = 11
            small_tensor[small_tensor < 10] = 10

            vmin, vmax = np.min(small_tensor), np.max(small_tensor)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(small_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            # ticks = [10,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11]
            ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
                     vmin + (vmax - vmin) / 10 * 4,
                     vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6, vmin + (vmax - vmin) / 10 * 7,
                     vmin + (vmax - vmin) / 10 * 8
                , vmin + (vmax - vmin) / 10 * 9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()
        elif i > 11 and i < 17:
            small_tensor = small_tensor*1000
            vmin, vmax = np.min(small_tensor), np.max(small_tensor)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(small_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
                     vmin + (vmax - vmin) / 10 * 4,
                     vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6, vmin + (vmax - vmin) / 10 * 7,
                     vmin + (vmax - vmin) / 10 * 8
                , vmin + (vmax - vmin) / 10 * 9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()
        else:
            vmin, vmax = np.min(small_tensor), np.max(small_tensor)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.contourf(small_tensor[0, 0], cmap='jet', levels=80, vmin=vmin, vmax=vmax)
            # 创建一个新的轴用于colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
            cbar = plt.colorbar(im, cax=cax)

            ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
                     vmin + (vmax - vmin) / 10 * 4,
                     vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6, vmin + (vmax - vmin) / 10 * 7,
                     vmin + (vmax - vmin) / 10 * 8
                , vmin + (vmax - vmin) / 10 * 9, vmax]
            cbar.set_ticks(ticks)

            file_name = f'{i + 1}.png'
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, bbox_inches='tight', dpi=500)
            plt.close()