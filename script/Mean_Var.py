# Time : 2023/8/30 20:59
# Tong ji Marcus
# FileName: Mean_Var.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list

# 图片文件夹路径
image_folder = r'E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\5\50\images'
# 获取文件夹中所有图片的文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]
# 初始化存储图像数据的列表
image_data = []

# 逐个读取图片并转换为矩阵后添加到列表中
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    # img_data = np.load(image_path)
    img_data = torch.tensor(np.load(image_path))
    img_data = img_data.view(1, 1, 128, 128)
    img_data = normalize(img_data)
    img_data = torch.where(img_data >= 0.5, torch.ones_like(img_data), torch.zeros_like(img_data))
    img_data = convert_to_np(img_data)
    img_data = np.squeeze(img_data, axis=0)
    image_data.append(img_data)

# 将列表转换为 NumPy 数组
image_data = np.array(image_data)

# 计算均值和方差
mean_image = np.mean(image_data, axis=0)
variance_image = np.var(image_data, axis=0)

# 找到方差的最大值，方便确定colorbar范围
print(np.max(variance_image))

# 绘制均值图像
plt.figure(figsize=(10, 5))
plt.imshow(mean_image, cmap='jet', vmin=0, vmax=1)
plt.colorbar()  # 添加colorbar
# plt.title('Mean Image')
plt.axis('off')  # 关闭坐标轴
plt.show()

# 绘制方差图像
plt.figure(figsize=(10, 5))
plt.imshow(variance_image, cmap='jet', vmin=0, vmax=0.25)
plt.colorbar()  # 添加colorbar
# plt.title('Variance Image')
plt.axis('off')  # 关闭坐标轴
plt.show()

