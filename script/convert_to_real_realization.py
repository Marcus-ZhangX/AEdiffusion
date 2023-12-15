# Time : 2023/7/12 21:33
# Tong ji Marcus
# FileName: convert_to_real_realization.py
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors

image = Image.open('E:/Coding_path/DiffuseVAE/scripts/reconstruction_samples/original/converted_TI_20000/crop15.jpg')  # 替换成你的图像路径
image = image.convert('L')
transform = T.Compose(
            [
                T.Resize((128, 128)),
                T.Grayscale(num_output_channels=1),  # 将图像转换为单通道
                T.ToTensor(),
                T.Lambda(lambda x: (x > 0.5).float()),  # 对图像进行二值化处理,原理是：如果不能满足x大于某个值的条件，就会将其变为黑色，即为灰度值为0
                # T.Normalize([0.5], [0.5]),    # 这个归一化操作会将像素值转化到[0,1]之间
            ]
        )
img = transform(image)
# 将img中大于等于0.5的像素值设置为1，小于0.5的像素值设置为0
img_binary = torch.where(img >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
# 转换为numpy数组并且去掉通道维度
img_data = img_binary.detach().cpu().numpy()
img_data = np.squeeze(img_data, axis=0)


###################################################################
# 保存为128x128大小的二值化JPEG图像
img_pil = Image.fromarray(np.uint8(img_data * 255))
img_pil.save('C:/Users/111/Desktop//11.jpg')

##################################################################
# 方式4
cmap = colors.ListedColormap(['black', 'white'])
plt.imshow(img_data, cmap=cmap)
plt.axis('off')  # 关闭坐标轴
plt.show()
##################################################################
# 方式2
img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
img_data = np.where(img_data > 0.5, 10, 0.4)
file_path = r'E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\real_cond_128.mat'
mat_data = {'real_cond_128': img_data}
sio.savemat(file_path, mat_data)

##################################################################
# 方式3
cmap = colors.ListedColormap(['black', 'white'])
bounds = [0.4, 4.8, 10]
norm = colors.BoundaryNorm(bounds, cmap.N)

# plt.imshow(img_data, cmap='Blues')  # jet
plt.imshow(img_data, cmap=cmap, norm=norm)
plt.colorbar(ticks=[0.4, 10], boundaries=bounds,shrink=0.5,aspect=10, fraction=0.1)
plt.axis('off')  # 关闭坐标轴
plt.show()



