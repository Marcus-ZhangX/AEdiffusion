# Time : 2023/6/4 23:56
# Tong ji Marcus
# FileName: gnerate_image_from_npy.py
import numpy as np
import matplotlib.pyplot as plt
import torch
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


# 画一张
n = 0
filename = f"E:/Coding_path/DiffuseVAE/scripts/ddpm_generated_samples/0/100/images/output_14epoch_0_3_{n}.npy"
# filename = f"E:/Coding_path/DiffuseVAE/scripts/reconstruction_samples/reconstructed/1000/images/output_14epoch_0_0_{n}.npy"

# style 1
img_data = torch.tensor(np.load(filename))
img_data = img_data.view(1, 1, 128, 128)
img_data = normalize(img_data)
img_data = torch.where(img_data >= 0.5, torch.ones_like(img_data), torch.zeros_like(img_data))
img_data = convert_to_np(img_data)
img_data = np.squeeze(img_data, axis=0)  # 将1*128*128*1的维度变为128*128*1
# print(img_data)

# style 2
# img_data = np.load(filename)  # 这里的imgdata不是非零即一的

plt.imshow(img_data, cmap='gray')
plt.axis('off')
plt.show()



#画一张50张图片的拼接大图
# 获取该文件夹下所有的.npy文件路径，并按名称排序
image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\5\50\images"
file_paths = sorted([os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])
# Choose the range of images you want to display (50 to 100 in this case)
start_index = 0
end_index = 50
file_paths = file_paths[start_index:end_index]

# 创建一个10列的图像显示窗口
n_rows = 5  # 2 5
n_cols = 10  # 8 10
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5))

# 读取每个.npy文件，进行预处理并显示在窗口中
for i, file_path in enumerate(file_paths):
    img_data = np.load(file_path)
    img_data = torch.tensor(np.load(file_path))
    img_data = img_data.view(1, 1, 128, 128)
    img_data = normalize(img_data)
    img_data = torch.where(img_data >= 0.5, torch.ones_like(img_data), torch.zeros_like(img_data))
    img_data = convert_to_np(img_data)
    img_data = np.squeeze(img_data, axis=0)
    row = i // n_cols
    col = i % n_cols
    # 将图片显示在子图中
    axes[row, col].imshow(img_data, cmap='gray')
    axes[row, col].axis('off')

# Hide any empty subplots
for i in range(len(file_paths), n_rows * n_cols):
    axes.flatten()[i].axis('off')

# Show the image grid
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

#===========================================================================================================
# 画16张reconstruction的图
# image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\Diffusevae_reconstruction\100\images"
image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\reconstructed\Diffusevae_reconstruction\100\images"
file_paths = sorted([os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])
# Choose the range of images you want to display (50 to 100 in this case)
start_index = 0
end_index = 16
file_paths = file_paths[start_index:end_index]
# 创建一个10列的图像显示窗口
n_rows = 4
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))

# 读取每个.npy文件，进行预处理并显示在窗口中
for i, file_path in enumerate(file_paths):
    img_data = np.load(file_path)
    img_data = torch.tensor(np.load(file_path))
    img_data = img_data.view(1, 1, 128, 128)
    img_data = normalize(img_data)
    img_data = torch.where(img_data >= 0.5, torch.ones_like(img_data), torch.zeros_like(img_data))
    img_data = convert_to_np(img_data)
    img_data = np.squeeze(img_data, axis=0)
    row = i // n_cols
    col = i % n_cols
    # 将图片显示在子图中
    axes[row, col].imshow(img_data, cmap='gray')
    axes[row, col].axis('off')

# Hide any empty subplots
for i in range(len(file_paths), n_rows * n_cols):
    axes.flatten()[i].axis('off')

# Show the image grid
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
#=============================================================================================================
#画反演5次之后的均值渗透系数场的图
image_folder_path =r"E:\Coding_path\DiffuseVAE\scripts\plot_results\50\images"
file_paths = sorted([os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])
# Choose the range of images you want to display (50 to 100 in this case)
start_index = 0
end_index = 5
file_paths = file_paths[start_index:end_index]

# 创建一个10列的图像显示窗口
n_rows = 2  # 2 5
n_cols = 3  # 8 10
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5))

# 读取每个.npy文件，进行预处理并显示在窗口中
for i, file_path in enumerate(file_paths):
    img_data = np.load(file_path)
    img_data = torch.tensor(np.load(file_path))
    img_data = img_data.view(1, 1, 128, 128)
    img_data = normalize(img_data)
    img_data = torch.where(img_data >= 0.5, torch.ones_like(img_data), torch.zeros_like(img_data))
    img_data = convert_to_np(img_data)
    img_data = np.squeeze(img_data, axis=0)
    row = i // n_cols
    col = i % n_cols
    # 将图片显示在子图中
    axes[row, col].imshow(img_data, cmap='gray')
    axes[row, col].axis('off')

# Hide any empty subplots
for i in range(len(file_paths), n_rows * n_cols):
    axes.flatten()[i].axis('off')

# Show the image grid
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()