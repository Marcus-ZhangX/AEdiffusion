# Time : 2023/10/5 19:20
# Tong ji Marcus
# FileName: calculate_PSNR_of_Kfileds.py
import os
from PIL import Image
import numpy as np


Ne = 800
# 加载参考图片
reference_image = Image.open("C:/Users/111/Desktop/reference_image.jpg")
# 转换为灰度图像
reference_image = reference_image.convert("L")
reference_array = np.array(reference_image)
reference_array = (reference_array > 128).astype(np.uint8)  # 将大于128的值设为1，小于等于128的值设为0

# 初始化PSNR总和
psnr_sum = 0

folder_path = r'E:\Coding_path\DiffuseVAE\scripts\ddpm_generated_samples\800Ne_5iter_50steps_range0.0001_with_surrogate\5\50\images'
for i in range(0, Ne):  # 将生成的K场写入cond_Ne.mat文件中
    k = i % 8  # 保证k的值始终在0、1、2、3之间循环
    batch_index = i // 8  # batch_index在0到4之间，而且会默认取整
    file_name = f'output_14epoch_0_{batch_index}_{k}.npy'
    file_path = os.path.join(folder_path, file_name)
    img_data = np.load(file_path)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    matrix = np.where(img_data > 0.5, 1, 0)
    # 计算均方误差
    mse = np.mean((reference_array - matrix) ** 2)
    # 计算PSNR
    psnr = 10 * np.log10(1 / mse)

    # 将PSNR添加到总和中
    psnr_sum += psnr

# 计算平均PSNR
average_psnr = psnr_sum / Ne

# 打印平均PSNR
print(f"Average PSNR: {average_psnr}")
