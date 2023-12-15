import os
import sys
import click
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(r"E:\Coding_path\DiffuseVAE\main")
from datasets.latent import UncondLatentDataset, VaeMyDataset
from models.vae import VAE
from util import configure_device, get_dataset, save_as_images
from pytorch_lightning.utilities.seed import seed_everything
#
# @click.group()
# def cli():
#     pass

#
# def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
#     # Plot all the quantities
#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
#     ax[0].imshow(gen.permute(1, 2, 0))
#     ax[0].set_title("VAE Reconstruction")
#     ax[0].axis("off")
#
#     ax[1].imshow(refined.permute(1, 2, 0))
#     ax[1].set_title("Refined Image")
#     ax[1].axis("off")
#
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, pad_inches=0)
#
#
# def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
#     N = len(interpolations)
#     # Plot all the quantities
#     fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)
#
#     for i, inter in enumerate(interpolations):
#         ax[i].imshow(inter.permute(1, 2, 0))
#         ax[i].axis("off")
#
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, pad_inches=0)
#
#
# def compare_interpolations(
#     interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
# ):
#     assert len(interpolations_1) == len(interpolations_2)
#     N = len(interpolations_1)
#     # Plot all the quantities
#     fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)
#
#     for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
#         ax[0, i].imshow(inter_1.permute(1, 2, 0))
#         ax[0, i].axis("off")
#
#         ax[1, i].imshow(inter_2.permute(1, 2, 0))
#         ax[1, i].axis("off")
#
#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, pad_inches=0)


#==========================================================================================================
# 这里是reconstruction的部分
parser = argparse.ArgumentParser(description='recons')
parser.add_argument('--chkpt_path', default=r"E:\Coding_path\DiffuseVAE\scripts\results_dir\vae_checkpoint_single_channel\30000_BS16_300epoch_256dim\checkpoints\vae-cmhq128_alpha=1.0-epoch=290-train_loss=0.0000-v1.ckpt", help='')
parser.add_argument('--root', default="E:/Coding_path/DiffuseVAE/converted_TI_20000", help='')  # E:/Coding_path/DiffuseVAE/scripts/reconstruction_samples/original
parser.add_argument('--device', default="gpu:0", help='')
parser.add_argument('--dataset', default="celebamaskhq", help='')
parser.add_argument('--image-size', default=128, help='')
parser.add_argument('--num_samples', default=16, help='')
parser.add_argument('--save-path', default=r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\only_vae_reconstruction", help='')
parser.add_argument("--write-mode", default="numpy", help='')
args = parser.parse_args(args=[])

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择cuda
if args.num_samples == 0:
    raise ValueError(f"`--num-samples` can take value=-1 or > 0")

# Dataset
dataset = get_dataset(args.dataset, args.root, args.image_size, norm=False, flip=False)
dataset = list(dataset)
dataset = dataset[:16]
# Loader
loader = DataLoader(
    dataset,
    batch_size=args.num_samples,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)

vae = VAE.load_from_checkpoint(args.chkpt_path, input_res=args.image_size) #  map_location="cpu"
vae = vae.cuda()
vae.eval()

sample_list = []
img_list = []
count = 0
for _, batch in tqdm(enumerate(loader)):
    batch = batch.cuda()  # 把数据移动到cuda上面去
    with torch.no_grad():
        recons = vae.forward_recons(batch)    # 生成样本的过程

    if count + recons.size(0) >= args.num_samples and args.num_samples != -1:
        img_list.append(batch[:args.num_samples, :, :, :].cpu())
        sample_list.append(recons[:args.num_samples, :, :, :].cpu())
        break

    # Not transferring to CPU leads to memory overflow in GPU!
    sample_list.append(recons.cpu())
    img_list.append(batch.cpu())
    count += recons.size(0)

cat_img = torch.cat(img_list, dim=0)
cat_sample = torch.cat(sample_list, dim=0)

# Save the image and reconstructions as numpy arrays
os.makedirs(args.save_path, exist_ok=True)

if args.write_mode == "image":
    save_as_images(
        cat_sample,
        file_name=os.path.join(args.save_path, "vae"),
        denorm=False,
    )
    save_as_images(
        cat_img,
        file_name=os.path.join(args.save_path, "orig"),
        denorm=False,
    )
else:
    np.save(os.path.join(args.save_path, "images.npy"), cat_img.numpy())
    np.save(os.path.join(args.save_path, "recons.npy"), cat_sample.numpy())
#===================================================================================================

# 将vae重建出来的图片画出来
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 读取.npy文件
data = np.load(r'E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\only_vae_reconstruction\recons.npy')
save_path = r'E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\only_vae_reconstruction'
# 查看数据
# print(data)
# print(data.shape)
# print(type(data))

# 循环生成图像
for i in range(16):
    img_data = data[i, :, :, :]
    img_data = np.squeeze(img_data, axis=0)

    # 归一化图像数据到0-1范围
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())

    # 二值化图像，使其成为灰度图
    img_data = np.where(img_data > 0.5, 1, 0)

    # 创建128x128像素的图像
    plt.figure(figsize=(1.28, 1.28), dpi=130)  # 指定图像大小和分辨率
    plt.imshow(img_data, cmap='gray')  # 绘制灰度图像
    plt.axis('off')  # 去除坐标轴

    # 保存图像
    plt.savefig(os.path.join(save_path, f'image_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=130)
    plt.close()


#===================================================================================================
# 这里是sample的部分
parser = argparse.ArgumentParser(description='sample')
parser.add_argument('--z_dim', default=256, help='')
parser.add_argument('--chkpt_path', default=r"E:/Coding_path/DiffuseVAE/scripts/results_dir/vae_checkpoint_single_channel/30000_BS16_300epoch_256dim/checkpoints/vae-cmhq128_alpha=1.0-epoch=290-train_loss=0.0000-v1.ckpt", help='')
parser.add_argument('--seed', default=0, help='')
parser.add_argument('--device', default="gpu:0", help='')
parser.add_argument('--image-size', default=128, help='')
parser.add_argument('--num_samples', default=16, help='')
parser.add_argument("--save-path", default=r"E:\Coding_path\DiffuseVAE\scripts\vae_generated_samples", help='')
parser.add_argument("--write-mode", default="numpy", help='')
args = parser.parse_args(args=[])
# print(args)
# seed_everything(args.seed)
# _, dev = configure_device(args.device)   # 可以仔细看看configure_device函数
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择cuda

# 导入潜在变量的两种方式：1.随机生成  2.通过文件z_a

# dataset = UncondLatentDataset((args.num_samples, args.z_dim, 1, 1))    # 进入UncondLatentDataset函数，发现里面也没有定义z_dim

data_dir = r"E:\Coding_path\DiffuseVAE\scripts\ILUES_vars"
dataset = VaeMyDataset(data_dir, args.num_samples)

# Loader
loader = DataLoader(
    dataset,
    batch_size=16,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)
# for i, x in enumerate(loader):
#     if i == 0:  # 注意索引从0开始
#         images = x  # 获取图像批次
#         break
# image = images[1]
# print(image)

vae = VAE.load_from_checkpoint(args.chkpt_path, input_res=args.image_size) #  map_location="cpu"
vae = vae.cuda()
vae.eval()

sample_list = []
count = 0
for _, batch in tqdm(enumerate(loader)):
    # print(batch)
    batch = batch.cuda()
    with torch.no_grad():
        recons = vae.forward(batch)
    if count + recons.size(0) >= args.num_samples and args.num_samples != -1:
        sample_list.append(recons[:args.num_samples, :, :, :].cpu())
        break

    # Not transferring to CPU leads to memory overflow in GPU!
    sample_list.append(recons.cpu())
    count += recons.size(0)

cat_sample = torch.cat(sample_list, dim=0)  #  它的作用是将一个列表（sample_list）中的张量沿着指定的维度（dim=0）进行拼接

# Save the image and reconstructions as numpy arrays
os.makedirs(args.save_path, exist_ok=True)

if args.write_mode == "image":
    save_as_images(
        cat_sample,   # 将生成的数组传入save_as_images
        file_name=os.path.join(args.save_path, "vae"),
        denorm=False,
    )
else:
    np.save(os.path.join(args.save_path, "sample.npy"), cat_sample.numpy())
#=============================================================================================================

# 将vae推理出来的图片画出来
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
# 读取.npy文件
data = np.load(r'E:\Coding_path\DiffuseVAE\scripts\vae_generated_samples\sample.npy')
save_path = r'E:\Coding_path\DiffuseVAE\scripts\vae_generated_samples'
# 查看数据
# print(data)
# print(data.shape)
# print(type(data))

for i in range(16):
    img_data = data[i, :, :, :]
    img_data = np.squeeze(img_data, axis=0)
    # print(img_data.shape)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    plt.imshow(img_data, cmap='gray')  # 绘制灰度图像
    plt.axis('off')  # 去除坐标轴
    plt.savefig(os.path.join(save_path, f'image_{i}.png'), bbox_inches='tight')
    plt.close()


