# Time : 2023/10/5 9:31
# Tong ji Marcus
# FileName: calculate_recons_R2.py
from PIL import Image
import numpy as np
# 读取两张图片
img1 = Image.open(r'C:\Users\111\Desktop\crop7.jpg').convert('L')  # 转换为灰度图像
img2 = Image.open(r'C:\Users\111\Desktop\image_7.png').convert('L')  # 转换为灰度图像

# 将图像转换为NumPy数组
img1_array = np.array(img1)
img2_array = np.array(img2)

# 计算R-Square指标
def r_square(img1, img2):
    # 将图像数据归一化到0-1范围
    img1_normalized = (img1 - img1.min()) / (img1.max() - img1.min())
    img2_normalized = (img2 - img2.min()) / (img2.max() - img2.min())

    # 计算差异
    diff = (img1_normalized - img2_normalized) ** 2

    # 计算R-Square
    r_square_value = 1 - np.sum(diff) / np.sum((img1_normalized - np.mean(img1_normalized))**2)

    return r_square_value

# 计算R-Square值
r_square_value = r_square(img1_array, img2_array)

# 将图像二值化为非零即一
threshold = 0.5  # 阈值，可以根据需要调整
binary_img = np.where(img1_array - img2_array > threshold, 1, 0)

# 打印R-Square值和二值化图像
print(f'R-Square值: {r_square_value}')

