import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import blobfile as bf
import matplotlib.pyplot as plt
import torchvision.transforms as T

# A very simplistic implementation of the CelebMaskHQ dataset supporting only images and no annotations
# TODO: Add functionality to download CelebA-MaskHQ and setup the dataset automatically
class CelebAMaskHQDataset(Dataset):
    def __init__(self, root, norm=False, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        # self.transform = transform
        self.norm = norm

        self.images = []

        img_path = os.path.join(self.root, "converted_TI_20000")
        image_filenames = os.listdir(img_path)
        sorted_image_filenames = sorted(image_filenames, key=lambda x: int(x[4:-4]))
        for img in tqdm(sorted_image_filenames):
            full_img_path = os.path.join(img_path, img)
            # print(full_img_path)
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
                # T.Normalize([0.5], [0.5]),    # 这个归一化操作会将像素值转化到[0,1]之间
            ]
        )
        img = transform(img)

        if self.norm:  # zx  改
            img = np.asarray(img).astype(np.float)  # img = (np.asarray(img).astype(np.float) / 127.5) - 1.0  # 使得图像的所有像素值都在 [-1, 1] 的范围内。
        else:
            img = np.asarray(img).astype(np.float)  # np.asarray(img).astype(np.float) / 255.0  # 使得图像的所有像素值都在 [0, 1] 的范围内。

        return torch.from_numpy(img).float()  # .permute(2, 0, 1)

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    root = r"E:\Coding_path\DiffuseVAE\converted_TI_20000"
    dataset = CelebAMaskHQDataset(root, subsample_size=None)
    print(len(dataset))



