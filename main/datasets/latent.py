import torch
from torch.utils.data import Dataset
from joblib import load
import scipy.io
import os
import numpy as np
# 这个文件中国只是定义了三个函数LatentDataset，UncondLatentDataset，ZipDataset
class LatentDataset(Dataset):
    def __init__(
        self,
        z_vae_size,  # 一个表示VAE（变分自编码器）潜在向量大小的整数或元组/列表。
        z_ddpm_size,  # 一个表示DDPM（概率扩散生成模型）潜在向量大小的整数或元组/列表。
        share_ddpm_latent=False,  # 一个布尔值，表示是否共享DDPM的潜在向量。
        expde_model_path=None,  # 一个字符串，表示Ex-PDE（扩散偏微分方程）模型的路径。
        **kwargs  # 额外的关键字参数。
    ):
        # NOTE: The batch index must be included in the latent code size input
        n_samples, *dims = z_ddpm_size  # 假设z_ddpm_size是一个包含多个元素的序列（如列表、元组等），例如z_ddpm_size = [100, 200, 300]，那么这段代码的作用是将第一个元素赋值给n_samples，而将剩余的元素赋值给dims。

        self.z_vae = torch.randn(z_vae_size)  # 使用torch.randn()函数生成一个大小为z_vae_size的随机张量，并将其赋值给self.z_vae
        self.share_ddpm_latent = share_ddpm_latent  # 是否共享DDPM的潜在向量

        # Load the Ex-PDE model and sample z_vae from it instead!
        if expde_model_path is not None and expde_model_path != "":  # 如果expde_model_path参数不为None且非空字符串，那么这段代码加载Ex-PDE模型并从中采样z_vae
            print("Found an Ex-PDE model. Will sample z_vae from it instead!")
            gmm = load(expde_model_path)
            gmm.set_params(random_state=kwargs.get("seed", 0))
            self.z_vae = (
                torch.from_numpy(gmm.sample(n_samples)[0]).view(z_vae_size).float()  # 使用了torch.from_numpy()将采样的潜在向量转换为PyTorch张量，并将其赋值给self.z_vae。
            )
            assert self.z_vae.size() == z_vae_size

        if self.share_ddpm_latent:  # 如果self.share_ddpm_latent为True，则使用torch.randn(dims)创建一个大小为dims的随机张量，并将其赋值给self.z_ddpm。
                                    # 如果self.share_ddpm_latent为False，则使用torch.randn(z_ddpm_size)创建一个大小为z_ddpm_size的随机张量，并将其赋值给self.z_ddpm。
            self.z_ddpm = torch.randn(dims)
        else:
            self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):  # 该函数定义了如何获取数据集中的单个样本，根据idx索引，它返回一个样本
        if self.share_ddpm_latent:
            return self.z_ddpm, self.z_vae[idx]  # self.z_ddpm（DDPM潜在向量）和self.z_vae[idx]（VAE潜在向量）作为一个元组返回
        return self.z_ddpm[idx], self.z_vae[idx]  # 分别返回self.z_ddpm[idx]（DDPM潜在向量）和self.z_vae[idx]（VAE潜在向量）。

    def __len__(self):  # __len__方法返回数据集的长度。它基于self.z_vae的大小，
        return int(self.z_vae.size(0))  # 使用size(0)方法返回第一个维度的长度，并将其转换为整数类型后返回。


class MyDatasetInitial(Dataset):
    def __init__(
        self,
        z_vae_size,  # 一个表示VAE（变分自编码器）潜在向量大小的整数或元组/列表。
        z_ddpm_size,  # 一个表示DDPM（概率扩散生成模型）潜在向量大小的整数或元组/列表。
        share_ddpm_latent=True,  # 一个布尔值，表示是否共享DDPM的潜在向量。
    ):
        # NOTE: The batch index must be included in the latent code size input
        n_samples, *dims = z_ddpm_size

        self.z_vae = torch.randn(z_vae_size)  # 使用torch.randn()函数生成一个大小为z_vae_size的随机张量，并将其赋值给self.z_vae
        self.share_ddpm_latent = share_ddpm_latent  # 是否共享DDPM的潜在向量

        if self.share_ddpm_latent:
            self.z_ddpm = torch.randn(dims)
        else:
            self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):  # 该函数定义了如何获取数据集中的单个样本，根据idx索引，它返回一个样本
        if self.share_ddpm_latent:
            return self.z_ddpm, self.z_vae[idx]  # self.z_ddpm（DDPM潜在向量）和self.z_vae[idx]（VAE潜在向量）作为一个元组返回
        return self.z_ddpm[idx], self.z_vae[idx]  # 分别返回self.z_ddpm[idx]（DDPM潜在向量）和self.z_vae[idx]（VAE潜在向量）。

    def __len__(self):  # __len__方法返回数据集的长度。它基于self.z_vae的大小，
        return int(self.z_vae.size(0))  # 使用size(0)方法返回第一个维度的长度，并将其转换为整数类型后返回。

class MyDatasetInversion(Dataset):
    def __init__(self, data_dir, Ne):
        self.data_dir = data_dir
        self.z_ddpm_file = "z_ddpm_128_128.mat"
        self.z_vae_file = "z_a.mat"
        self.Ne = Ne
        # Load z_ddpm from mat file
        z_ddpm_file = os.path.join(self.data_dir, self.z_ddpm_file)
        z_ddpm_data = scipy.io.loadmat(z_ddpm_file)["z_ddpm"]  # mat文件的变量名字"z_ddpm"
        # z_ddpm_data = np.tile(z_ddpm_data, (self.Ne, 1, 1))  # self.Ne是要生成的图片的数量，这里会复制self.Ne个z_ddpm
        z_ddpm_data = np.expand_dims(z_ddpm_data, axis=0)  # 在第 1 维度上添加一个维度
        self.z_ddpm = torch.from_numpy(z_ddpm_data).float()

        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_a"]  # mat文件的变量名字"z_a"
        z_vae_data = np.split(z_vae_data, self.Ne, axis=1)  # 在第 1 维度上切割为 self.Ne 个张量
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]  # 在第 1、2 维度上添加维度
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):

        return self.z_ddpm, self.z_vae[idx]


class MyDatasetResults(Dataset):
    def __init__(self, data_dir, iter):
        self.data_dir = data_dir
        self.z_ddpm_file = "z_ddpm_128_128.mat"
        self.z_vae_file = "z_mean_results.mat"
        self.iter = iter
        # Load z_ddpm from mat file
        z_ddpm_file = os.path.join(self.data_dir, self.z_ddpm_file)
        z_ddpm_data = scipy.io.loadmat(z_ddpm_file)["z_ddpm"]  # mat文件的变量名字"z_ddpm"
        # z_ddpm_data = np.tile(z_ddpm_data, (self.iter+1, 1, 1))  # self.Ne是要生成的图片的数量，这里会复制self.Ne个z_ddpm
        z_ddpm_data = np.expand_dims(z_ddpm_data, axis=0)  # 在第 1 维度上添加一个维度
        self.z_ddpm = torch.from_numpy(z_ddpm_data).float()

        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_mean_results"]  # mat文件的变量名字"z_a"
        z_vae_data = np.split(z_vae_data, self.iter+1, axis=1)  # 在第 1 维度上切割为 self.Ne 个张量
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]  # 在第 2 维度上添加维度
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):

        return self.z_ddpm, self.z_vae[idx]

class UncondLatentDataset(Dataset):
    def __init__(self, z_ddpm_size, **kwargs):
        # NOTE: The batch index must be included in the latent code size input
        self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):
        return self.z_ddpm[idx]

    def __len__(self):
        return int(self.z_ddpm.size(0))

class VaeMyDataset(Dataset):
    def __init__(self, data_dir, Ne):
        self.data_dir = data_dir
        self.z_vae_file = "z_a.mat"
        self.Ne = Ne
        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_a"]  # mat文件的变量名字"z_a"
        z_vae_data = np.split(z_vae_data, self.Ne, axis=1)  # 在第 1 维度上切割为 self.Ne 个张量
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]  # 在第 1、2 维度上添加维度
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):
        return self.z_vae[idx]

class ZipDataset(Dataset):
    def __init__(self, recons_dataset, latent_dataset, **kwargs):
        # NOTE: The batch index must be included in the latent code size input
        assert len(recons_dataset) == len(latent_dataset)
        self.recons_dataset = recons_dataset
        self.latent_dataset = latent_dataset

    def __getitem__(self, idx):
        return self.recons_dataset[idx], self.latent_dataset[idx]

    def __len__(self):
        return len(self.recons_dataset)
