# Helper script to sample from a conditional DDPM model
# Add project directory to sys.path
import os
import sys
# p = os.path.join(os.path.abspath("."), "main")
# sys.path.insert(1, p)
import copy
import hydra
import pytorch_lightning as pl
import torch
import numpy as np
import scipy.io

print(os.getcwd())  # 显示当前路径
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")
from datasets.latent import MyDatasetInitial, MyDatasetInversion,MyDatasetResults # 替换原来的LatentDataset
from models.callbacks import ImageWriter
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel
from models.vae import VAE
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from util import configure_device


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(version_base=None, config_path=r"E:\Coding_path\DiffuseVAE\main\configs")
def sample_cond(config):
    # Seed and setup
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)  # 默认 0

    batch_size = config_ddpm.evaluation.batch_size  # 生成图片的组数8
    n_steps = config_ddpm.evaluation.n_steps  # 这里的n_steps是可以变化的，对应推理过程中的步长，可以与训练过程中的步长不相等
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size  # 图片大小 128
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path  # 这里的ddpm_latent_path应当为 “ ”，为空的话后面才能使得share_ddpm_latent=True
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None
    # 关于这里latent path参数的选择。我想如果只是使用z_vae作为条件去生成样本的话，那么这里应该就是空的路径。
    # 如果需要加入一些有意义的词语进行控制的话，比如胡子，性别，表情。那么这里就是这些东西  的保存路径

    # Load pretrained VAE
    vae = VAE.load_from_checkpoint(
        config_vae.evaluation.chkpt_path,
        input_res=image_size,
    )
    vae.eval()

    # Load pretrained wrapper
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)  # "32,16,8"
    dim_mults = __parse_str(config_ddpm.model.dim_mults)  # "1,2,2,3,4"
    decoder = SuperResModel(
        in_channels=config_ddpm.data.n_channels,  # 1
        model_channels=config_ddpm.model.dim,  # 128
        out_channels=1,
        num_res_blocks=config_ddpm.model.n_residual,  # 2
        attention_resolutions=attn_resolutions,  # "32,16,8"
        channel_mult=dim_mults,  # "1,2,2,3,4"
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,  # 0.0
        num_heads=config_ddpm.model.n_heads,  # 1
        z_dim=config_ddpm.evaluation.z_dim,  # 100
        use_scale_shift_norm=config_ddpm.evaluation.z_cond,  # 理论上来讲是True
        use_z=config_ddpm.evaluation.z_cond,  # 理论上来讲是True
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,   # 1000
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )

    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,  # 模型保存路径
        online_network=online_ddpm,
        target_network=target_ddpm,
        vae=vae,             # 在sample方法里，这里是none
        conditional=True,
        pred_steps=n_steps,   # 这里的n_steps是可以变化的，对应推理过程中的步长，可以与训练过程中的步长不相等
        eval_mode="sample",   # 只有 "sample", "recons"这两种模式
        resample_strategy=config_ddpm.evaluation.resample_strategy,  # "spaced"
        skip_strategy=config_ddpm.evaluation.skip_strategy,  # "uniform"
        sample_method=config_ddpm.evaluation.sample_method,  # "ddpm"
        sample_from=config_ddpm.evaluation.sample_from,  # "target"
        data_norm=config_ddpm.data.norm,  # False
        temp=config_ddpm.evaluation.temp,  # 1.0 看过代码了，这个参数应该影响不大
        guidance_weight=config_ddpm.evaluation.guidance_weight,  # 0.0  这个参数需要调参
        z_cond=config_ddpm.evaluation.z_cond,  # True
        strict=False,   # 原 True /这个语句就是指忽略掉模型和参数文件中不匹配的参数
        ddpm_latents=ddpm_latents,  # 传入DDPMWrapper，传入spaced diffussion,在spaced_diff中的192行
    )

    # Create predict dataset of latents
    # z_dataset = LatentDataset(
    #     (n_samples, config_vae.model.z_dim, 1, 1),  # 这个n_samples=10， z_dim=100
    #     (n_samples, 1, image_size, image_size),  # image_size = 128
    #     share_ddpm_latent=True if ddpm_latent_path != "" else False,   # 分享ddpm过程中的所有随机性，这样能使得生成结果的随机性更小
    #     expde_model_path=config_vae.evaluation.expde_model_path,
    #     seed=config_ddpm.evaluation.seed,  # 这个seed，不是用来防止数据被打乱的，而是用在expde_model_path的判断语句内部的
    # )
    if decision == 0:
        z_dataset = MyDatasetInitial(
            (n_samples, config_vae.model.z_dim, 1, 1),  # 这个n_samples=10， z_dim=100
            (n_samples, 1, image_size, image_size),  # image_size = 128
            share_ddpm_latent=True if ddpm_latent_path != "" else False,  # 分享ddpm过程中的所有随机性，这样能使得生成结果的随机性更小
        )
        # z_dataset = MyDatasetInitial(
        #     (10, 100, 1, 1),  # 这个n_samples=10， z_dim=100
        #     (10, 1, 128, 128),  # image_size = 128
        # )
        # 顺便将z_vae写入路径文件中
        idx_list = range(len(z_dataset))
        z_vae_column = torch.stack([z_dataset[idx][1] for idx in idx_list], dim=1)
        z_vae_column = z_vae_column.squeeze().cpu().numpy()
        z_vae_Ne = np.zeros((config_vae.model.z_dim, n_samples))  # (100,10)  (config_vae.model.z_dim, n_samples)
        # print(z_vae_column[:, 1])
        # print(z_vae_column[:, 0].shape)
        for i in range(z_vae_column.shape[1]):
            z_vae_Ne[:, i] = z_vae_column[:, i]
        file_path1 = r'E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\z_vae_100.mat'
        data1 = {"z_vae_100": z_vae_Ne}   # "z_vae_100"是保存z_vae_Ne时的名字，保存的值为z_vae_Ne
        scipy.io.savemat(file_path1, data1)

        # 顺便将z_ddpm写入路径文件中
        z_ddpm_array = z_dataset[0][0]  # 获取第一个样本的 z_ddpm
        # print(z_ddpm_array.shape)  # torch.Size([1, 128, 128])
        z_ddpm_array = z_ddpm_array.squeeze().cpu().numpy()  # 去除不必要的维度
        # print(z_ddpm_array.shape)  # (128, 128)
        # print(type(z_ddpm_array))  # <class 'numpy.ndarray'>
        z_ddpm_array = z_ddpm_array.astype(np.double)
        file_path2 = r"E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\z_ddpm_128_128.mat"
        data2 = {"z_ddpm": z_ddpm_array}  # "z_ddpm"是保存z_ddpm_array时的名字，保存的值为z_ddpm_array
        scipy.io.savemat(file_path2, data2)

        # val_loader = DataLoader(
        #     z_dataset,
        #     batch_size=5,
        #     drop_last=False,  # 不要drop last 。如果一共n_samples=10，如果我设置batch_size=8，drop_last=True的话，那么就只剩八张图了
        #     shuffle=False,  # 不打乱数据
        #     )
        # for i, (batch,_) in enumerate(val_loader):  # 获得ddpm的dataset
        #     if i == 0:
        #         images = batch
        #         break
        # for i, (_,batch) in enumerate(val_loader):  # 获得z_vae的dataset
        #     if i == 1:
        #         images = batch
        #         break
        # # 获取第10张图像
        # print(type(images))
        # print(type(images))
        # image = images[0]
        # print(image.shape)
        # print(image

    elif decision == 1:  # 这里要写一个函数用来导入每一次反演之后得到的z
        data_dir = r"E:\Coding_path\DiffuseVAE\scripts\ILUES_vars"
        z_dataset = MyDatasetInversion(data_dir, n_samples)
        # val_loader = DataLoader(
        #     z_dataset,
        #     batch_size=5,
        #     drop_last=False,  # 不要drop last 。如果一共n_samples=10，如果我设置batch_size=8，drop_last=True的话，那么就只剩八张图了
        #     shuffle=False,  # 不打乱数据
        #     )
        # for i, (batch,_) in enumerate(val_loader):  # 获得ddpm的dataset
        #     if i == 0:
        #         images = batch
        #         break
        # for i, (_,batch) in enumerate(val_loader):  # 获得z_vae的dataset
        #     if i == 0:
        #         images = batch
        #         break
        # # 获取第10张图像
        # print(type(images))
        # print(images.shape)
        # image = images[4,:,:,:]
        # print(image.shape)
        # print(image)

    elif decision == 2:  # 这里要写一个函数用来导入反演结束之后的z_mean_results
        data_dir = r"E:\Coding_path\DiffuseVAE\scripts\ILUES_vars"
        z_dataset = MyDatasetResults(data_dir, config_ddpm.evaluation.iter)

        # Setup devices
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True  # 默认是false，如果设置了大于0，那么就必须是true
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    # Predict loader
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,  # 不要drop last 。如果一共n_samples=10，如果我设置batch_size=8，drop_last=True的话，那么就只剩八张图了
        pin_memory=True,
        shuffle=False,  # 不打乱数据
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )

    # Predict trainer
    if decision == 0:
        save_path = os.path.join(config_ddpm.evaluation.save_path, "0")
    elif decision == 1:
        save_path = os.path.join(config_ddpm.evaluation.save_path, str(m))
    elif decision == 2:
        save_path = r"E:\Coding_path\DiffuseVAE\scripts\plot_results"

    write_callback = ImageWriter(
        save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=True,  # 这个参数设置为True的话，最后的prediction会被分成两部分 ddpm_samples_dict, vae_samples = prediction
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_vae=config_ddpm.evaluation.save_vae,
        # save_mode=config_ddpm.evaluation.save_mode,
        is_norm=True,  # 按照save as np方式生成的图片是否采用normalize操作。选择True
    )

    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)


if __name__ == "__main__":
    import time
    import numpy as np
    import matlab.engine
    import scipy.io
    import os
    import datetime
    import time
    import torch

    import argparse
    import torch.nn.functional as F

    import torch
    import torch.nn as nn
    import sys
    import os
    # from torchsummary import summary

    import matplotlib.pyplot as plt
    import random
    import scipy.io
    import numpy as np
    from scipy.io import savemat
    eng = matlab.engine.start_matlab('MATLAB_R2021a')  # 定义matlab.engine

    # def clear():  # 释放内存/ 不能释放的变量目前有：xall,start_time,end_time
    #     for key, value in globals().items():
    #         if callable(value) or value.__class__.__name__ == "module":
    #             continue
    #         del globals()[key]

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


    parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
    parser.add_argument('--exp-name', type=str, default='AR-Net', help='experiment name')
    parser.add_argument('--blocks', type=list, default=(5, 10, 5),
                        help='list of number of layers in each block in decoding net')  # 解码网络中每一块的层数列表
    parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')  # 每层卷积的输出
    parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
    parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                        help='enable bottleneck in the dense blocks')
    parser.add_argument('--init-features', type=int, default=48,
                        help='# initial features after the first conv layer')  # 初始卷积层输出的特征面数量
    args = parser.parse_args(args=[])

    model = DenseED(3, 2, blocks=args.blocks, growth_rate=args.growth_rate,
                    drop_rate=args.drop_rate, bn_size=args.bn_size,
                    num_init_features=args.init_features, bottleneck=args.bottleneck).cuda()

    model_dir = r"E:\Coding_path\DiffuseVAE\scripts"
    model.load_state_dict(torch.load(model_dir + '/model_epoch{}.pth'.format(200)))
    print('Loaded model')

#-------------------------------------------------------------------------------------------------------
    def get_simv(y_pred):
        # # One can also use the built-in Rbf interpolation of Python,
        # # but here we use the interpolation function in MATLAB instead, as it is about 10 times faster
        # f = Rbf(x, y, y_output,  function='multiquadric')
        # y_sim = f(xobs, yobs)
        scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/y_pred.mat', dict(y_pred=y_pred))
        eng.interp_matlab(nargout=0)
        y_sim = np.loadtxt("y_sim.dat")
        return y_sim

    def normalize(obj):
        # 获取输入张量的维度信息
        B, C, H, W = obj.shape
        # 遍历所有通道（这里只有一个通道）
        for i in range(1):
            # 提取当前通道的所有像素值并展平
            channel_val = obj[:, i, :, :].view(B, -1)
            # 对当前通道进行最小-最大规范化
            channel_val -= channel_val.min(1, keepdim=True)[0]
            channel_val /= (channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0])
            # 将处理后的像素值变形成与输入张量相同的形状
            channel_val = channel_val.view(B, H, W)
            # 用处理后的像素值替换原始张量中当前通道的像素值
            obj[:, i, :, :] = channel_val
        # 返回处理后的张量
        return obj

    def Postprocess(file_path):
        img_data = torch.tensor(np.load(file_path))
        img_data = img_data.view(1, 1, 128, 128)
        img_data = normalize(img_data)
        b1 = torch.full(img_data[0][0].shape, 10.0)  # 创建一个形状与output[0][0]相同的张量b1，其中所有元素均为6，并将其移动到GPU上。
        b2 = torch.full(img_data[0][0].shape, 0.4)  # 创建一个形状与output[0][0]相同的张量b2，其中所有元素均为2，并将其移动到GPU上。
        img_data = torch.where(img_data >= 0.5, b1, b2)  # 对应通道处的渗透系数10，非通道处为0.4（m/d）
        img_data = img_data.detach().cpu().numpy()
        img_data = np.squeeze(img_data, axis=0)  # 将1*128*128*1的维度变为128*128*1
        return img_data

    def gene_ss_and_save(range_vals, N=1, file_path='E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\ss_200.mat'):
        if N is None:
            N = 1
        Npar = range_vals.shape[0]
        x = np.empty((Npar, N))
        for i in range(N):
            x[:, i] = range_vals[:, 0] + (range_vals[:, 1] - range_vals[:, 0]) * np.random.rand(Npar)
        savemat(file_path, {'ss_Ne': x})
#=============================================================================================================
    # 开始初始化Ne个realization并且运行替代模型，得到y1
    start_time1 = time.time()  # 开始时间
    Ne = 600
    ngx = 128
    ngy = 128
    Nt = 5
    Nobs = 25 * 5 + 25  # 观测井数量

    # 生成ss_Ne
    range_vals = np.array([[0, 100], [100, 200], [200, 300], [200, 300], [300, 400]])
    gene_ss_and_save(range_vals, N=Ne)
    # 生成Ne个K场
    decision = 0  # 初始化，使得导入初始化随机z_dim和z_ddpm
    sample_cond()

    # 将ss_Ne和z_vae合并成为x1
    mat_data = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/ILUES_vars/z_vae_100.mat')
    z_vae_100 = mat_data['z_vae_100']
    mat_data2 = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/ILUES_vars/ss_200.mat')
    ss_Ne = mat_data2['ss_Ne']
    # print(ss_Ne)
    # print(ss_Ne.shape)
    # print(type(ss_Ne))
    x1 = np.vstack((z_vae_100, ss_Ne))
    scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/x1.mat', {'x1': x1})
    # 初始化y1
    y1 = np.zeros((Nobs, Ne))

    model.eval()
    folder_path = r'E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\50\images'
    for i in range(0, Ne):  # 将生成的K场写入cond_Ne.mat文件中
        k = i % 8  # 保证k的值始终在0、1、2、3之间循环
        batch_index = i//8  # batch_index在0到4之间，而且会默认取整
        file_name = f'output_14epoch_0_{batch_index}_{k}.npy'
        file_path = os.path.join(folder_path, file_name)
        output = Postprocess(file_path)  # 将图片进行后处理，主要实现让其变为非零即一
        output = np.squeeze(output)  # 压缩多余的维度
        output = np.log(output)
        # print(output)
        # print((output.shape))  # (1, 128, 128)
        # print(type(output))
        # output = output.squeeze()
        Sx_id = 11  # 第12列  11
        source = np.full((Nt, 128, 128), 0.0)
        for j in range(Nt):
            for p in range(33, 93):  # 33, 93
                Sy_id = p
                source[j, Sy_id, Sx_id] = ss_Ne[j, i]
        # scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/source.mat', dict(source=source))

        x = np.full((1, 3, 128, 128), 0.0)  # three input channels: hydraulic conductivity field, source term, previous concentration field
        y = np.full((Nt, 2, 128, 128), 0.0)  # two output channles: concentration and head fields
        y_i_1 = np.full((128, 128), 0.0)  # y_0 = 0
        for q in range(Nt):
            x[0, 0, :, :] = output  # hydraulic conductivity
            x[0, 1, :, :] = source[q]  # source rate
            x[0, 2, :, :] = y_i_1  # the i-1)^th predicted concentration field, which is treated as an input channel
            x_tensor = (torch.FloatTensor(x)).cuda()
            with torch.no_grad():
                y_hat = model(x_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y[q] = y_hat  # y_hat的大小为[1,2,128,128]
            y_i_1 = y_hat[0, 0, :, :]  # the updated (i-1)^th predicted concentration field

        y_pred = np.full((Nt + 1, 128, 128), 0.0)
        y[:, 0] *= 1000  # 训练模型的训练数据将浓度值除了1000，这里要乘回来
        y_pred[:Nt] = y[:, 0]  # the concentration fields at Nt time instances

        column_to_modify = y[0, 1]
        # 将大于11的值设置为11，小于10的值设置为10
        column_to_modify[column_to_modify > 11] = 11
        column_to_modify[column_to_modify < 10] = 10
        # 更新修改后的列回到原始矩阵
        y[0, 1] = column_to_modify

        y_pred[Nt] = y[0, 1]  # the hydraulic head field
        # print(y_pred)
        # print(y_pred.shape) # (6, 128, 128)
        # print(type(y_pred))

        # 得到y_pred的shape是（5+1，128，128），需要把观测点处的conc和head拿出来，作为观测值
        # 获得观测点处的坐标
        y1[:, i] = get_simv(y_pred)  # 这里传入的y_pred大小为[6,128,128]

    scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/y1.mat', {'conc_head_Ne': y1})
    end_time1 = time.time()  # 结束时间
    execution_time1 = end_time1 - start_time1
    print("Execution Time of Inversion:", execution_time1)
# -----------------------------------------------------------------------------------------------------
# 开始反演
    x_para = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/x1.mat')  # z_Ne.mat cat ss_Ne.mat
    xf = x_para['x1']
    obs_model = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/y1.mat')
    yf = obs_model['conc_head_Ne']

    scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/xf.mat', {"xf": xf})  # Initial ensemble
    scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/yf.mat', {"yf": yf})
    xall = xf
    yall = yf

    start_time2 = time.time()  # 开始时间
    # 开始反演
    m = 0
    N_iter = 5
    ngx = 128
    ngy = 128
    Nobs = 150
    Nt = 5
    Ne = 600
    # 初始化ya,观测值
    ya = np.zeros((Nobs, Ne))
    model.eval()

    eng.cd('E:/Coding_path/DiffuseVAE/scripts', nargout=0)  # 将matlab的engine的路径调整到路径下

    for i in range(0, N_iter):
        m += 1  # m是用来保存每一次的生成的Ne张图片的
        eng.ilues1(nargout=0)  # 执行反演算法
        xa = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/xa.mat')  # xa是已经更新过的ss和z_latent
        xa = xa['xa']

        # 每次反演迭代时，写入时间戳
        print('iter=', i + 1)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 按照指定格式获取当前时间戳
        file_path = r"E:\Coding_path\DiffuseVAE\scripts\timestamp.txt"
        with open(file_path, 'a') as file:
            content = f"iter {i+1} done: {timestamp}\n"  # 构造写入的内容字符串
            file.write(content)  # 写入文件

        z_a = xa[0:256, :]  # 0到256行的所有列，待修改，这里导入z_a的方式需要改变，如何写道dataset中，主要问题是导入的变量对象不明确
        ss_a = xa[256:, :]  # 256行以后得所有列

        file_path4 = r'E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\z_a.mat'
        data4 = {"z_a": z_a}
        scipy.io.savemat(file_path4, data4)

        file_path5 = r'E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\ss_a.mat'
        data5 = {"ss_a": ss_a}
        scipy.io.savemat(file_path5, data5)

        decision = 1  # 使得导入反演后的z_vae
        sample_cond()  # 执行推理

        folder_path = rf'E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\{i+1}\50\images'
        for j in range(0, Ne):  # 将生成的K场写入cond_Ne.mat文件中
            k = j % 8  # 保证k的值始终在0、1、2、3之间循环
            batch_index = j // 8  # batch_index在0到4之间，而且会默认取整
            file_name = f'output_14epoch_0_{batch_index}_{k}.npy'
            file_path = os.path.join(folder_path, file_name)
            output = Postprocess(file_path)  # 将图片进行后处理，主要实现让其变为非零即一，然后再赋值为10.0和0.4
            output = np.squeeze(output)  # 压缩多余的维度
            output = np.log(output)
            # print(output)
            # print((output.shape))  # (128, 128, 1)
            Sx_id = 11  # 第12列
            source = np.full((Nt, 128, 128), 0.0)
            for j in range(Nt):
                for p in range(33, 93):
                    Sy_id = p
                    source[j, Sy_id, Sx_id] = ss_a[j, i]

            x = np.full((1, 3, 128, 128),0.0)  # three input channels: hydraulic conductivity field, source term, previous concentration field
            y = np.full((Nt, 2, 128, 128), 0.0)  # two output channles: concentration and head fields
            y_i_1 = np.full((128, 128), 0.0)  # y_0 = 0
            for q in range(Nt):
                x[0, 0, :, :] = output  # hydraulic conductivity
                x[0, 1, :, :] = source[q]  # source rate
                x[0, 2, :, :] = y_i_1  # the i-1)^th predicted concentration field, which is treated as an input channel
                x_tensor = (torch.FloatTensor(x)).cuda()
                with torch.no_grad():
                    y_hat = model(x_tensor)
                y_hat = y_hat.data.cpu().numpy()
                y[q] = y_hat
                y_i_1 = y_hat[0, 0, :, :]  # the updated (i-1)^th predicted concentration field

            y_pred = np.full((Nt + 1, 128, 128), 0.0)
            y[:, 0] *= 1000
            y_pred[:Nt] = y[:, 0]   # the concentration fields at Nt time instances

            column_to_modify = y[0, 1]
            # 将大于11的值设置为11，小于10的值设置为10
            column_to_modify[column_to_modify > 11] = 11
            column_to_modify[column_to_modify < 10] = 10
            # 更新修改后的列回到原始矩阵
            y[0, 1] = column_to_modify

            y_pred[Nt] = y[0, 1]  # the hydraulic head field

            # 得到y_pred的shape是（5+1，128，128），需要把观测点处的conc和head拿出来，作为观测值
            # 获得观测点处的坐标
            ya[:, i] = get_simv(y_pred)

        scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/ya.mat', {"ya": ya})

        # eng.forward_model(nargout=0)  # 获得ss_a所对应的观测值
        eng.update_samples(nargout=0)  # accept or reject the candidate?  把观测点的数据提取出来

        xa = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/xa.mat')  # The updated inputs
        ya = scipy.io.loadmat('E:/Coding_path/DiffuseVAE/scripts/ya.mat')  # The updated outputs
        xa = xa['xa']
        ya = ya['ya']

        xall = np.concatenate((xall, xa), axis=1)  # 在列方向上拼接两个数组， xall 保存所有每一次迭代的结果
        yall = np.concatenate((yall, ya), axis=1)

    scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/results.mat', {"xall": xall, "yall": yall})  # save results

    end_time2 = time.time()  # 结束时间
    execution_time2 = end_time2 - start_time2
    print("Execution Time of Inversion:", execution_time2)
##===========================================================================================================
# 通过matlab求得反演得到的每一次迭代之后Ne个z_vae的均值，得到的是一个256*iter大小的矩阵，通过如下代码推理得到图片

    # decision = 2  # 使得导入z_mean_results
    # sample_cond()  # 执行推理
#==========================================================================================================
    # 用来实验相同潜在变量生成图片相似性的语句
    # m = 0
    # decision = 1  # 使得导入反演后的z_vae
    # start_time = time.time()  # 开始时间
    # sample_cond()  # 执行推理
    # end_time = time.time()  # 结束时间
    # execution_time2 = end_time - start_time
    # print("Execution Time of Inversion:", execution_time2)

#==========================================================================================================
    # folder_path = r'E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\50\images'
    # file_name = f'output_14epoch_0_0_5.npy'
    # file_path = os.path.join(folder_path, file_name)
    # output = Postprocess(file_path)  # 将图片进行后处理，主要实现让其变为非零即一
    # output = np.squeeze(output)
    # # output = np.log(output)
    # scipy.io.savemat('E:/Coding_path/DiffuseVAE/scripts/output.mat', dict(output=output))