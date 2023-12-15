# Time : 2023/10/5 21:28
# Tong ji Marcus
# FileName: plot_posterior_Kfiled_transport.py
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors

image = Image.open('C:/Users/111/Desktop/reference_image.jpg')  # 替换成你的图像路径
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

img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
img_data = np.where(img_data > 0.5, 10, 0.4)
file_path = r'E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\real_cond_128.mat'
mat_data = {'real_cond_128': img_data}
sio.savemat(file_path, mat_data)
#######################################################################################
#计算后验均值场，并且保存为mat文件
import numpy as np
import torch
import os
from scipy.io import savemat  # 导入保存.mat文件的函数
import matplotlib.pyplot as plt
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
    img_data = torch.where(img_data >= 0.5, 1.0, 0.0)
    img_data = convert_to_np(img_data)
    img_data = np.squeeze(img_data, axis=0)
    image_data.append(img_data)

# 将列表转换为 NumPy 数组
image_data = np.array(image_data)

# 计算均值和方差
posterior_128_cond = np.mean(image_data, axis=0)
# 将大于0.5的值替换为10，小于0.5的值替换为0.4
posterior_128_cond = np.where(posterior_128_cond > 0.5, 10, 0.4)
# 保存mean_image为.mat文件
save_path = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts\posterior_128_cond_surrogate.mat'
savemat(save_path, {'posterior_128_cond': posterior_128_cond})

############################################################################

#画出各个后验的污染羽图
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 载入.mat文件
mat_file = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts\posterior_transport\posterior_1000d.mat'  # 替换成实际的.mat文件路径
# mat_file = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts\reference_transport\reference_1000d.mat'  # 替换成实际的.mat文件路径
mat_data = sio.loadmat(mat_file)
small_tensor = mat_data['posterior_1000d']  # 替换成.mat文件中矩阵的名称
small_tensor[small_tensor < 0] = 0
# 对small_tensor进行处理
scaled_tensor = np.log10(small_tensor + 1)
vmin, vmax = np.min(scaled_tensor), np.max(scaled_tensor)

# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.contourf(scaled_tensor, cmap='jet', levels=80, vmin=vmin, vmax=vmax)

# 创建一个新的轴用于colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
cbar = plt.colorbar(im, cax=cax)

# 设置colorbar的刻度
ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
         vmin + (vmax - vmin) / 10 * 4, vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6,
         vmin + (vmax - vmin) / 10 * 7, vmin + (vmax - vmin) / 10 * 8, vmin + (vmax - vmin) / 10 * 9, vmax]
cbar.set_ticks(ticks)

# 保存图像
save_path = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts'
file_name = f'output_image.png'
file_path = os.path.join(save_path, file_name)
plt.savefig(file_path, bbox_inches='tight', dpi=500)
plt.close()


##############################################################################################
#画出reference和posterior的残差图

#画出各个后验的污染羽图
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 载入.mat文件
mat_file1 = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts\reference_transport\reference_1000d.mat'  # 替换成实际的.mat文件路径
mat_file2 = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts\posterior_transport_surrogate\posterior_1000d_surrogate.mat'  # 替换成实际的.mat文件路径
mat_data1 = sio.loadmat(mat_file1)
mat_data2 = sio.loadmat(mat_file2)
small_tensor1 = mat_data1['reference_1000d']  # 替换成.mat文件中矩阵的名称
small_tensor2 = mat_data2['posterior_1000d_surrogate']
small_tensor1[small_tensor1 < 0] = 0
small_tensor2[small_tensor2 < 0] = 0
scaled_tensor1 = np.log10(small_tensor1 + 1)
scaled_tensor2 = np.log10(small_tensor2 + 1)
scaled_tensor = scaled_tensor1-scaled_tensor2
#########################################################################
# mat_data = sio.loadmat(mat_file2)
# small_tensor = mat_data['posterior_1000d_surrogate']
# small_tensor[small_tensor < 0] = 0
# scaled_tensor = np.log10(small_tensor + 1)

# 对small_tensor进行处理
vmin, vmax = np.min(scaled_tensor), np.max(scaled_tensor)
# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.contourf(scaled_tensor, cmap='jet', levels=80, vmin=vmin, vmax=vmax)

# 创建一个新的轴用于colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.4)  # 调整size和pad以控制colorbar的大小和位置
cbar = plt.colorbar(im, cax=cax)

# 设置colorbar的刻度
ticks = [vmin, vmin + (vmax - vmin) / 10, vmin + (vmax - vmin) / 10 * 2, vmin + (vmax - vmin) / 10 * 3,
         vmin + (vmax - vmin) / 10 * 4, vmin + (vmax - vmin) / 10 * 5, vmin + (vmax - vmin) / 10 * 6,
         vmin + (vmax - vmin) / 10 * 7, vmin + (vmax - vmin) / 10 * 8, vmin + (vmax - vmin) / 10 * 9, vmax]
cbar.set_ticks(ticks)

# 保存图像
save_path = r'E:\Coding_path\DiffuseVAE\scripts\reference_and_posterior_transport_resluts'
file_name = f'output_image.png'
file_path = os.path.join(save_path, file_name)
plt.savefig(file_path, bbox_inches='tight', dpi=500)
plt.close()