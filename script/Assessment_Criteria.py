# Time : 2023/7/30 13:16
# Tong ji Marcus
# FileName: Assessment_Criteria.py
import os
import numpy as np

def SamplesDifference():
    '''
    该参数是用来衡量相同潜在变量生成的50张图片之间的相似度的，寻找guidance weight 参数时用到了
    '''
    images = []
    image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\50\images"
    file_paths = sorted([os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])
    # Choose the range of images you want to display (50 to 100 in this case)
    start_index = 0
    end_index = 50
    file_paths = file_paths[start_index:end_index]

    for i, file_path in enumerate(file_paths):
        img_data = np.load(file_path)
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        img_data = np.where(img_data > 0.5, 1, 0)
        images.append(img_data)
    # 计算像素平均值矩阵
    mean_matrix = np.mean(np.array(images), axis=0)
    # 初始化累加差值的矩阵
    diff_sum = np.zeros(mean_matrix.shape)
    # 计算每张图片和平均矩阵的差值并累加
    for image in images:
        diff = np.abs(image - mean_matrix)
        diff_sum += diff
    # 计算指标
    metric = np.sum(diff_sum) / (len(images)*128*128)
    print("差异指标值:", metric)

SamplesDifference()


#=======================================================================================================
# 计算图片的黑白占比
# 方式1：计算npy文件的黑白占比
import os
import numpy as np

# 文件夹路径
folder_path = r"E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\100\images"

# 初始化黑白像素计数器
total_black_pixels = 0
total_white_pixels = 0

# 遍历文件夹下的所有npy文件
for npy_filename in os.listdir(folder_path):
    npy_path = os.path.join(folder_path, npy_filename)
    # 读取npy文件
    img = np.load(npy_path)
    # print(img.shape)
    img_data = np.squeeze(img)
    # print(img_data.shape)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.45, 1, 0)
    # 统计黑白像素
    total_black_pixels += np.sum(img_data == 0)
    total_white_pixels += np.sum(img_data == 1)

# 计算黑白像素所占比例
total_pixels = img.size * 1000
black_pixel_percentage = (total_black_pixels / total_pixels) * 100
white_pixel_percentage = (total_white_pixels / total_pixels) * 100

print("黑色像素所占比例：{:.2f}%".format(black_pixel_percentage))
print("白色像素所占比例：{:.2f}%".format(white_pixel_percentage))

# 方式2：计算图片文件的黑白占比

from PIL import Image
import os
# 文件夹路径
folder_path = r"E:\Coding_path\DiffuseVAE\converted_TI_20000\converted_TI_20000"
# 初始化黑白像素计数器
total_black_pixels = 0
total_white_pixels = 0

# 遍历文件夹下的所有图片
for img_filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_filename)
    # 打开图片
    img = Image.open(img_path).convert("L")
    img_data = np.array(img)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    # 统计黑白像素
    total_black_pixels += np.sum(img_data == 0)
    total_white_pixels += np.sum(img_data == 1)
# 计算黑白像素所占比例
total_pixels = img.width * img.height * 20000
black_pixel_percentage = (total_black_pixels / total_pixels) * 100
white_pixel_percentage = (total_white_pixels / total_pixels) * 100

print("黑色像素所占比例：{:.2f}%".format(black_pixel_percentage))
print("白色像素所占比例：{:.2f}%".format(white_pixel_percentage))

#----------------------------------------------------------------------------------------------
#画出训练集黑白占比的柱状图
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# 图像文件夹路径
image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\100\images"

# 获取文件列表
file_paths = sorted(
    [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])

# 初始化存储像素统计结果的列表
pixel_counts_0 = []
pixel_counts_1 = []

# 遍历每个.npy文件并进行统计
for file_path in file_paths:
    img_data = np.load(file_path)
    # print(img.shape)
    img_data = np.squeeze(img_data )
    # print(img_data.shape)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    # 统计0和1的像素个数
    count_0 = np.sum(img_data == 0)
    count_1 = np.sum(img_data == 1)

    # 将统计结果添加到列表中
    pixel_counts_0.append(count_0)
    pixel_counts_1.append(count_1)

# 计算每个图像中0和1的像素百分比
total_pixels = 128 * 128  # 图像总像素数
percentages_0 = [(count / total_pixels) * 100 for count in pixel_counts_0]
percentages_1 = [(count / total_pixels) * 100 for count in pixel_counts_1]

# 创建柱状图
x = np.arange(len(file_paths))  # x轴标签位置
width = 0.4  # 柱子宽度

fig, ax = plt.subplots(figsize=(4, 6))
cmap_jet = get_cmap("jet")
# 选择"jet"颜色映射中的第一个颜色
color_1 = cmap_jet(25)
color_2 = cmap_jet(1000)
rects1 = ax.bar(x - width / 2, percentages_0, width, label='0', color=color_1, linestyle='--')
rects2 = ax.bar(x + width / 2, percentages_1, width, label='1', color=color_2, linestyle='-.')


# 设置图表标题和标签
ax.set_ylabel('Percentage')
ax.set_xlabel('training data')
ax.set_xticks([])

# 显示图表
plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------
#画出生成图像的黑白占比的柱状图
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# 文件夹路径
folder_path = r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\original\converted_TI_20000"
# 初始化黑白像素计数器
pixel_counts_0 = []
pixel_counts_1 = []

# 遍历文件夹下的所有图片
for img_filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_filename)
    # 打开图片
    img = Image.open(img_path).convert("L")
    img_data = np.array(img)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    count_0 = np.sum(img_data == 0)
    count_1 = np.sum(img_data == 1)
    # 将统计结果添加到列表中
    pixel_counts_0.append(count_0)
    pixel_counts_1.append(count_1)

# 计算每个图像中0和1的像素百分比
total_pixels = 128 * 128  # 图像总像素数
percentages_0 = [(count / total_pixels) * 100 for count in pixel_counts_0]
percentages_1 = [(count / total_pixels) * 100 for count in pixel_counts_1]

# 创建柱状图
x = np.arange(16)  # x轴标签位置
width = 0.4  # 柱子宽度

fig, ax = plt.subplots(figsize=(4, 6))
cmap_jet = get_cmap("jet")
# 选择"jet"颜色映射中的第一个颜色
color_1 = cmap_jet(25)
color_2 = cmap_jet(1000)
rects1 = ax.bar(x - width / 2, percentages_0, width, label='0', color=color_1, hatch='///', edgecolor='black')
rects2 = ax.bar(x + width / 2, percentages_1, width, label='1', color=color_2, hatch='///', edgecolor='black')


# 设置图表标题和标签
ax.set_ylabel('Percentage')
ax.set_xlabel('Random realizations')
ax.set_xticks([])

# 显示图表
plt.tight_layout()
plt.show()




#-------------------------------------------------------------------------------
# 画双x轴图的代码
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# 图像文件夹路径
image_folder_path = r"E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\0\100\images"

# 获取文件列表
file_paths = sorted(
    [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith('.npy')])

# 初始化存储像素统计结果的列表
pixel_counts_0 = []
pixel_counts_1 = []

# 遍历每个.npy文件并进行统计
for file_path in file_paths:
    img_data = np.load(file_path)
    # print(img.shape)
    img_data = np.squeeze(img_data )
    # print(img_data.shape)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    # 统计0和1的像素个数
    count_0 = np.sum(img_data == 0)
    count_1 = np.sum(img_data == 1)

    # 将统计结果添加到列表中
    pixel_counts_0.append(count_0)
    pixel_counts_1.append(count_1)

# 计算每个图像中0和1的像素百分比
total_pixels = 128 * 128  # 图像总像素数
percentages_0 = [(count / total_pixels) * 100 for count in pixel_counts_0]
# 取反percentages_1以使柱子朝下增长
percentages_1 = [-1 * (count / total_pixels) * 100 for count in pixel_counts_1]

# 创建柱状图
x = np.arange(16)  # x轴标签位置
width = 0.9  # 柱子宽度

fig, ax = plt.subplots(figsize=(4, 6))
color_1 = (95 / 255, 151 / 255, 210 / 255)  # RGB值(95, 151, 210)
color_2 = (215 / 255, 99 / 255, 100 / 255)  # RGB值(215, 99, 100)
rects1 = ax.bar(x, percentages_0, width, label='0', color=color_1, edgecolor='black')

# 创建第二个X轴
ax2 = ax.twiny()
# 使用取反后的percentages_1来使柱子朝下增长
rects2 = ax2.bar(x, percentages_1, width, label='1', color=color_2, edgecolor='black', bottom=100)

# 设置图表标题和标签
ax.set_ylabel('Percentage')
ax.set_xlabel('Random realizations')
ax.set_xticks([])

# 配置第二个X轴
ax2.set_xlabel('Random realizations')
ax2.set_xticks([])

# 设置y轴范围为0-100
ax.set_ylim(0, 100)

# 显示图表
plt.tight_layout()
plt.show()


from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# 文件夹路径
folder_path = r"E:\Coding_path\DiffuseVAE\scripts\reconstruction_samples\original\converted_TI_20000"

# 初始化黑白像素计数器
pixel_counts_0 = []
pixel_counts_1 = []

# 遍历文件夹下的所有图片
for img_filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_filename)
    # 打开图片
    img = Image.open(img_path).convert("L")
    img_data = np.array(img)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    img_data = np.where(img_data > 0.5, 1, 0)
    count_0 = np.sum(img_data == 0)
    count_1 = np.sum(img_data == 1)
    # 将统计结果添加到列表中
    pixel_counts_0.append(count_0)
    pixel_counts_1.append(count_1)

# 计算每个图像中0和1的像素百分比
total_pixels = 128 * 128  # 图像总像素数
percentages_0 = [(count / total_pixels) * 100 for count in pixel_counts_0]
# 取反percentages_1以使柱子朝下增长
percentages_1 = [-1 * (count / total_pixels) * 100 for count in pixel_counts_1]

# 创建柱状图
x = np.arange(16)  # x轴标签位置
width = 0.9 # 柱子宽度

fig, ax = plt.subplots(figsize=(4, 6))

color_1 = (95 / 255, 151 / 255, 210 / 255)  # RGB值(95, 151, 210)
color_2 = (215 / 255, 99 / 255, 100 / 255)  # RGB值(215, 99, 100)

rects1 = ax.bar(x, percentages_0, width, label='0', color=color_1, hatch='///', edgecolor='black')

# 创建第二个X轴
ax2 = ax.twiny()
# 使用取反后的percentages_1来使柱子朝下增长
rects2 = ax2.bar(x, percentages_1, width, label='1', color=color_2, hatch='///', edgecolor='black', bottom=100)

# 设置图表标题和标签
ax.set_ylabel('Percentage')
ax.set_xlabel('Random realizations')
ax.set_xticks([])

# 配置第二个X轴
ax2.set_xlabel('Random realizations')
ax2.set_xticks([])

# 设置y轴范围为0-100
ax.set_ylim(0, 100)

# 显示图表
plt.tight_layout()
plt.show()





