# Time : 2023/6/5 0:49
# Tong ji Marcus
# FileName: experiment.py
import torch
from PIL import Image
import numpy as np


def binary_image(x):

    x = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    x = torch.where(x >= 0.5, torch.ones_like(x), torch.zeros_like(x))

    return x

x = torch.randn(2, 1, 128, 128)

# 将张量变为非零即一的黑白灰度图，并将其二值化
x_binary = binary_image(x)

# 输出处理后的张量
print(x_binary.shape)  # 输出: torch.Size([2, 1, 128, 128])
print(x_binary)        # 输出: 包含0和1两个值的张量


###################
import torch
from torchvision import transforms
from PIL import Image
def image_to_tensor(filename):
    # 定义图像的变换，包括调整大小和归一化操作
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    # 加载图像文件并应用变换
    img = Image.open(filename)
    img = np.asarray(img).astype(np.float) / 255.0
    # 返回结果张量
    return img
tensor = image_to_tensor('E:\Coding_path\DiffuseVAE\converted_TI_20000\converted_TI_20000\crop3.jpg')
print(tensor)


#######################
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.Grayscale(num_output_channels=1),  # 将图像转换为单通道
                T.ToTensor(),
                T.Lambda(lambda x: (x > 0.5).float()),  # 对图像进行二值化处理,原理是：如果不能满足x大于某个值的条件，就会将其变为黑色，即为灰度值为0
                # T.Normalize([0.5], [0.5]),    # 这个归一化操作会将像素值转化到[0,1]之间
            ]
        )
img = Image.open('E:\Coding_path\DiffuseVAE\converted_TI_20000\converted_TI_20000\crop3.jpg')
img = transform(img)
print(img)
img1 = np.asarray(img).squeeze()
plt.imshow(img1)



import torchvision.transforms as transform
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
img0=Image.open(r'C:\Users\111\Desktop\crop1102.jpg')
# img =np.array(img0)
# print(img.shape)
img1=transform.Grayscale(num_output_channels=1)(img0)
img = transform.ToTensor()(img0)
img = transform.Lambda(lambda x: (x > 0.5).float())(img)
img1 = np.asarray(img).squeeze()

axs = plt.figure().subplots(1, 2)
axs[0].imshow(img0);axs[0].set_title('src');axs[0].axis('off')
axs[1].imshow(img1,cmap=plt.get_cmap('gray'));axs[1].set_title('gray');axs[1].axis('off')
plt.show()

#############################################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader

class CelebAMaskHQDataset(Dataset):
    def __init__(self, root, norm=False, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        # self.transform = transform
        self.norm = norm

        self.images = []

        img_path = os.path.join(self.root, "converted_TI_30000")
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
root = "E:/Coding_path/DiffuseVAE/converted_TI_30000"
dataset = CelebAMaskHQDataset(root, norm=False)



loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,    # 按照官方的建议[3]是你默认设置pin_memory为True就对了
        shuffle=True,
        drop_last=True,
)

for i, batch in enumerate(loader):  # 获得z_vae的dataset
    if i == 0:
        images = batch
        break
# 获取第10张图像
print(type(images))
print(images.shape)
image = images[4,:,:,:]
print(image.shape)
print(image)
img = np.asarray(image).squeeze()
plt.imshow(img)


