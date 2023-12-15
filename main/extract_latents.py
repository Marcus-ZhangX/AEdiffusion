import argparse
import sys
import scipy.io

sys.path.append(r"E:\Coding_path\DiffuseVAE\main")
from torch.utils.data import DataLoader
from models.vae import VAE
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description='recons')
parser.add_argument('--vae-chkpt-path', default=r"E:\Coding_path\DiffuseVAE\scripts\results_dir\vae_checkpoint_single_channel\30000_BS16_300epoch_256dim\checkpoints\vae-cmhq128_alpha=1.0-epoch=290-train_loss=0.0000-v1.ckpt", help='')
parser.add_argument('--root', default="E:/Coding_path/DiffuseVAE/scripts/extract_reference_latent_samples_16", help='')
parser.add_argument('--device', default="gpu:0", help='')
parser.add_argument('--dataset', default="celebamaskhq", help='')
parser.add_argument('--image_size', default=128, help='')
parser.add_argument('--save_path', default="E:/Coding_path/DiffuseVAE/scripts/z_latent_64.mat", help='')
args = parser.parse_args(args=[])


class CelebAMaskHQDataset2(Dataset):
    def __init__(self, root, subsample_size=None):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.images = []

        img_path = self.root
        image_filenames = os.listdir(img_path)
        for img in tqdm(image_filenames):
            full_img_path = os.path.join(img_path, img)
            self.images.append(full_img_path)
        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.Grayscale(num_output_channels=1),  # 将图像转换为单通道
                T.ToTensor(),
                T.Lambda(lambda x: (x > 0.5).float()),  # 对图像进行二值化处理,原理是：如果不能满足x大于某个值的条件，就会将其变为黑色，即为灰度值为0
            ]
        )
        img = transform(img)
        img = np.asarray(img)
        return torch.from_numpy(img).float()

    def __len__(self):
        return len(self.images)


# dev, _ = configure_device(device)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择cuda
# Dataset
dataset = CelebAMaskHQDataset2(args.root, subsample_size=16)
# Loader
loader = DataLoader(
    dataset,
    batch_size=16,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)
# for i, batch in enumerate(loader):  # 获得ddpm的dataset
#     if i == 0:
#         images = batch
#         break
# print(type(images))
# print(images.shape)
# image = images[4,:,:,:]
# print(image.shape)
# print(image)
# img = np.asarray(image).squeeze()
# plt.imshow(img)


# Load VAE
vae = VAE.load_from_checkpoint(args.vae_chkpt_path, input_res=args.image_size).to(dev)
vae.eval()

z_list = []
for _, batch in tqdm(enumerate(loader)):
    batch = batch.cuda()
    with torch.no_grad():
        mu, logvar = vae.encode(batch)
        z = vae.reparameterize(mu, logvar)

    # Not transferring to CPU leads to memory overflow in GPU!
    z_list.append(z.cpu())

z_latent = torch.cat(z_list, dim=0)
z_latent = np.array(z_latent)
# Save the latents as numpy array
file_path = args.save_path
data = {"z_latent": z_latent}
scipy.io.savemat(file_path, data)

#==============================================================================================
# 画出z的直方图
import matplotlib.pyplot as plt
import scipy.io as sio
# Load the .mat file
mat_file = 'E:/Coding_path/DiffuseVAE/scripts/z_latent_256.mat'
# Load the data from the first row of the .mat file
data = sio.loadmat(mat_file)['z_latent']
# Extract the first row (256 values)
first_row_data = data[0, :, 0, 0]
# Create a histogram
plt.hist(first_row_data, bins=12, color='k', alpha=0.7, rwidth=0.85)
# Show the histogram
plt.yticks([])
plt.show()
