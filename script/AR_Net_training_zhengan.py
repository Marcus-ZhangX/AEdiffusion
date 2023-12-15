# Time : 2023/7/4 12:32
# Tong ji Marcus
# FileName: AR_Net_training_zhengan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
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

# 以上都不用管
#-----------------------------------------------------------------------------------------------
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import h5py
import os
import argparse
import matplotlib.pyplot as plt
from torchinfo import summary

plt.switch_backend('agg')  # 用于输出图像

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

# parser.add_argument('--w-c', type=float, default=8.0, help='weight value in the weighted loss')
# parser.add_argument('--w-c2', type=float, default=4.0, help='weight value in the weighted loss')
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

#---------------------------------------------------------------------------------

all_over_again = 'train_ARNet_result'

# 试验数据存放路径
exp_dir = args.data_dir + all_over_again + "/{}/Ntrs{}__Bks{}_Bts{}_Eps{}_wd{}_lr{}_K{}_TBts{}_growth_rate{}inverse". \
    format(args.exp_name, args.n_train, args.blocks,
           args.batch_size, args.n_epochs, args.weight_decay, args.lr, args.growth_rate, args.test_batch_size,
           args.growth_rate)
# 输出数据存放路径

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#---------------------------------------------------------------------------------
# 开始读取训练数据，转换为hdf5文件
import time
import numpy as np
import scipy.io
import h5py
def read_input_sourceIndex_train(ndata, ntimes, Nt, ngx, ngy):
    # ntimes is the total number of time instances considered
    # Nt is the number of time instances with non-zero source rate
    x = np.full((ndata * ntimes, 3, ngx, ngy), 0.0)  # three input channels: (K, r_i, y_{i-1})
    y = np.full((ndata * ntimes, 2, ngx, ngy), 0.0)  # one output channels: y_i 输出是污染物浓度值和水头值
    k = 0
    cond_surrogate = scipy.io.loadmat("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/input_training/cond_surrogate.mat")
    cond_surrogate1 = cond_surrogate['cond_surrogate']
    #数据测试
    # K = cond_surrogate1[:, 0]
    # K = np.log(K)
    # K = K.reshape(128, 128)
    # plt.imshow(K, cmap='gray')  # 使用灰度颜色映射
    # plt.show()
    for i in range(1, ndata + 1):
        K = cond_surrogate1[:, i-1]
        K = np.log(K)  # 取对数，使得原始的10,0.4之间的差距更小
        K = K.reshape(ngx, ngy)

        source = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/input_training/ss{}.dat".format(i))

        head = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/output_training/head_{}.dat".format(i))  # 水头参数文件（输出
        # source = np.loadtxt("E:/Coding_path/DiffuseVAE/scripts/Surrogate_model/input_training/ss1.dat")
        S_rate = source[0:]  # 污染物释放速率
        y_j_1 = np.full((ngx, ngy), 0.0)  # y_0 = 0
        for j in range(1, ntimes + 1):
            x[k, 0, :, :] = K  # K is the first input channel
            if j <= Nt:
                Sx_id = 11  # 第12列
                for p in range(33, 93):
                    Sy_id = p
                    x[k, 1, Sy_id, Sx_id] = S_rate[j - 1]
            x[k, 2, :, :] = y_j_1  # the (j-1)th output is the third output channel

            y_j = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/output_training/conc_{}_t{}.dat".format(i, j))
            y_j = np.where(y_j > 0, y_j, 0.0)
            y_j = y_j/1000  # 除以1000


            y[k, 0, :, :] = y_j  # the jth output
            y[k, 1, :, :] = head  # head is the second output channel
            y_j_1 = y_j
            k += 1

    print("x: {}".format(x[:ntimes, :]))
    print("y_c: {}".format(y[:ntimes, 0]))
    print("y_h: {}".format(y[ntimes, 1]))
    print("Ymax: {}".format(np.max(y[:, :, :, :])))

    hf = h5py.File('E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/input_lhs{}_train.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data=x, dtype='f', compression='gzip')
    hf.close()  # x表示输入，三个通道放到一个h5py里面

    hf = h5py.File('E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/output_lhs{}_train.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data=y, dtype='f', compression='gzip')
    hf.close()  # y表示输出。


start_time = time.time()  # 开始时间

ndata = 3000
ntimes = 5
Nt = 5  # the number of time instances with non-zero source rate
ngx = 128
ngy = 128
read_input_sourceIndex_train(ndata, ntimes, Nt, ngx, ngy)

end_time = time.time()  # 结束时间
execution_time = end_time - start_time
print("Execution Time:", execution_time)
# #---------------------------------------------------------------------------------

def read_input_sourceIndex_test(ndata, ntimes, Nt, ngx, ngy):
    # ntimes is the total number of time instances considered
    # Nt is the number of time instances with non-zero source rate
    x1 = np.full((ndata * ntimes, 3, ngx, ngy), 0.0)  # three input channels: (K, r_i, y_{i-1})
    y1 = np.full((ndata * ntimes, 2, ngx, ngy), 0.0)  # one output channels: y_i 输出是污染物浓度值和水头值
    k = 0
    cond_surrogate = scipy.io.loadmat("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/input_testing/cond_surrogate_test.mat")
    cond_surrogate1 = cond_surrogate['cond_surrogate_test']
    #数据测试
    # K = cond_surrogate1[:, 0]
    # K = K.reshape(128, 128)
    # plt.imshow(K, cmap='gray')  # 使用灰度颜色映射
    # plt.show()
    for i in range(1, ndata + 1):
        K1 = cond_surrogate1[:, i-1]
        K1 = np.log(K1)  # 取对数，使得原始的10,0.4之间的差距更小
        K1 = K1.reshape(ngx, ngy)
        source1 = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/input_testing/ss{}.dat".format(i))

        head1 = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/output_testing/head_{}.dat".format(i))  # 水头参数文件（输出
        # source = np.loadtxt("E:/Coding_path/DiffuseVAE/scripts/Surrogate_model/input_training/ss1.dat")
        S_rate = source1[0:]  # 污染物释放速率

        y_j_11 = np.full((ngx, ngy), 0.0)  # y_0 = 0
        for j in range(1, ntimes + 1):
            x1[k, 0, :, :] = K1  # K is the first input channel
            if j <= Nt:
                Sx_id = 11  # 第12列
                for p in range(33, 93):
                    Sy_id = p
                    x1[k, 1, Sy_id, Sx_id] = S_rate[j - 1]
            x1[k, 2, :, :] = y_j_11  # the (j-1)th output is the third output channel

            y_j = np.loadtxt("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/output_testing/conc_{}_t{}.dat".format(i, j))
            y_j = np.where(y_j > 0, y_j, 0.0)
            y_j = y_j / 1000  # 除以1000


            y1[k, 0, :, :] = y_j  # the jth output
            y1[k, 1, :, :] = head1  # head is the second output channel
            y_j_11 = y_j
            k += 1

    print("x1: {}".format(x1[:ntimes, :]))
    print("y_c: {}".format(y1[:ntimes, 0]))
    print("y_h: {}".format(y1[ntimes, 1]))
    print("Ymax: {}".format(np.max(y1[:, :, :, :])))

    hf = h5py.File('E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/input_lhs{}_testing.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data=x1, dtype='f', compression='gzip')
    hf.close()  # x表示输入，三个通道放到一个h5py里面

    hf = h5py.File('E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/output_lhs{}_testing.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data=y1, dtype='f', compression='gzip')
    hf.close()  # y表示输出。

start_time = time.time()  # 开始时间

ndata = 500
ntimes = 5
Nt = 5  # the number of time instances with non-zero source rate
ngx = 128
ngy = 128
read_input_sourceIndex_test(ndata, ntimes, Nt, ngx, ngy)

end_time = time.time()  # 结束时间
execution_time = end_time - start_time
print("Execution Time:", execution_time)
#---------------------------------------------------------------------------------
import h5py
import netron as netron
ntimes = 5
Nt = 5  # the number of time instances with non-zero source rate
ngx = 128
ngy = 128
# 读取hdf5文件
with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/input_lhs{}_train.hdf5".format(args.n_train), 'r') as f:
    x_train = f['dataset'][()]
    print("train input data shape: {}".format(x_train.shape))
with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/training/output_lhs{}_train.hdf5".format(args.n_train), 'r') as f:
    y_train = f['dataset'][()]
    print("train output data shape: {}".format(y_train.shape))

with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/input_lhs{}_testing.hdf5".format(args.n_test), 'r') as f:
    x_test = f['dataset'][()]
    print("test input data shape: {}".format(x_test.shape))
with h5py.File("E:/data_and_code/data/data_for_ARnet/Surrogate_model/testing/output_lhs{}_testing.hdf5".format(args.n_test), 'r') as f:
    y_test = f['dataset'][()]
    print("test output data shape: {}".format(y_test.shape))

#---------------------------------------------------------------------------------

# 计算训练数据均值方差并储存在stats
y_train_mean = np.mean(y_train, 0)
y_train_var = np.sum((y_train - y_train_mean) ** 2)
print('y_train_var: {}'.format(y_train_var))
train_stats = {}
train_stats['y_mean'] = y_train_mean
train_stats['y_var'] = y_train_var
# 计算测试数据均值方差并储存在stats
y_test_mean = np.mean(y_test, 0)
y_test_var = np.sum((y_test - y_test_mean) ** 2)
print('y_test_var: {}'.format(y_test_var))
test_stats = {}
test_stats['y_mean'] = y_test_mean
test_stats['y_var'] = y_test_var
# num_workers：使用多进程加载的进程数（和CPU核数有关），0代表不使用多进程；pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
kwargs = {'num_workers': 0,
          'pin_memory': True} if th.cuda.is_available() else {}


# data_train打包训练数据的输入输出，测试数据的输入输出。DataLoader负责加载数据
data_train = th.utils.data.TensorDataset(th.FloatTensor(x_train),
                                         th.FloatTensor(y_train))
data_test = th.utils.data.TensorDataset(th.FloatTensor(x_test),
                                        th.FloatTensor(y_test))
train_loader = th.utils.data.DataLoader(data_train,
                                        batch_size=args.batch_size,
                                        shuffle=True, **kwargs)
test_loader = th.utils.data.DataLoader(data_test,
                                       batch_size=args.test_batch_size,
                                       shuffle=True, **kwargs)

model = DenseED(x_train.shape[1], y_train.shape[1], blocks=args.blocks, growth_rate=args.growth_rate,
                drop_rate=args.drop_rate, bn_size=args.bn_size,
                num_init_features=args.init_features, bottleneck=args.bottleneck).to(device)
print(model)
print("number of parameters: {}\nnumber of layers: {}"
      .format(*model._num_parameters_convlayers()))

summary(model, input_size=(1, 3, 128, 128))

# 利用netron 展示网络结构的可视化图
# data = torch.randn(1, 3, 128, 128).cuda()  # 随机生成一个输入
# torch.onnx.export(
#     model,
#     data,
#     'E:/data_and_code/data/data_for_ARnet/Surrogate_model/ARnet网络结构可视化/model.onnx',
#     export_params=True,
#     opset_version=11,
# )

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer, mode = 'min', factor = 0.1, patience = 10,
    verbose = True, threshold = 0.0001, threshold_mode = 'rel',
    cooldown = 0, min_lr = 0, eps = 1e-08)  # 改进学习率函数

n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()

#--------------------------------------------------------------------------------------------

# 定义test函数
def test(epoch, plot_intv):
    model.eval()
    loss = 0.
    for batch_idx, (input, target) in enumerate(
            test_loader):  # test_loader有x和y两层，inout表示x，target表示y，将x输入model，得到的结果与y进行比较
        input, target = input.to(device), target.to(device)
        with th.no_grad():
            output = model(input)
        loss += F.mse_loss(output, target, size_average=False).item()

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    r2_score = 1 - loss / test_stats['y_var']
    print("epoch: {}, test r2-score:  {:.4f}".format(epoch, r2_score))
    print("epoch: {}, test rmse:  {:.4f}".format(epoch, rmse_test))
    return r2_score, rmse_test

#------------------------------------------------------------------------------------------------
# 开始训练
import time
start_time = time.time()  # 开始时间

R2_test_self = []
r2_train, r2_test = [], []
rmse_train, rmse_test = [], []

for epoch in range(1, args.n_epochs + 1):
    # train
    model.train()
    mse = 0.
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)

        loss = F.l1_loss(output, target, reduction='sum')

        # for computing the RMSE criterion solely
        loss_mse = F.mse_loss(output, target, reduction='sum')
        # loss_mse = mean_squared_error(output, target)
        loss.backward()
        optimizer.step()
        mse += loss_mse.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / train_stats['y_var']
        print("epoch: {}, training mse: {}".format(epoch, mse))
        print("epoch: {}, training r2-score: {:.6f}".format(epoch, r2_score))
        print("epoch: {}, training rmse: {:.6f}".format(epoch, rmse))
        r2_train.append(r2_score)

        rmse_train.append(rmse)

        r2_t, rmse_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)
        print("loss: {}".format(loss))

    scheduler.step(rmse)

    # save model
    if epoch % 2 == 0:
        th.save(model.state_dict(), model_dir + "/model_epoch{}.pth".format(epoch))

end_time = time.time()  # 结束时间
execution_time = end_time - start_time

print("Done training {} epochs with {} data using {} seconds".format(args.n_epochs, args.n_train, execution_time))

#---------------------------------------------------------------------------------
# 训练后的结果的参数，绘图
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimSun'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

plt.figure()
x = np.arange(args.log_interval, args.n_epochs + args.log_interval, args.log_interval)
plt.plot(x, r2_train, 'k', label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
plt.plot(x, r2_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('$R^2$', fontsize=14)
plt.legend(loc='lower right')
plt.savefig(exp_dir + "/r2.png", dpi=400)
plt.close()

np.savetxt(exp_dir + "/r2_train.txt", r2_train)
np.savetxt(exp_dir + "/r2_test.txt", r2_test)

plt.figure()
plt.plot(x, rmse_train, 'k', label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
plt.plot(x, rmse_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(loc='upper right')
plt.savefig(exp_dir + "/rmse.png", dpi=400)
plt.close()
np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)



