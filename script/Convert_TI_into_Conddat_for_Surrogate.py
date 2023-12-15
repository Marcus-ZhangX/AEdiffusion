# Time : 2023/9/7 13:56
# Tong ji Marcus
# FileName: Convert_TI_into_Conddat_for_Surrogate.py
import os
import scipy.io
import numpy as np
from PIL import Image

def Postprocess(file_path):
    img_data = Image.open(file_path)
    img_data = np.array(img_data)
    img_data = img_data/255
    img_data = np.where(img_data > 0.5, 10.0, 0.4)
    return img_data

Ne = 500
ngx = 128
ngy = 128
cond_surrogate_test = np.zeros(((ngx * ngy), Ne))
folder_path = r'E:\Coding_path\DiffuseVAE\converted_TI_30000\converted_TI_30000'
for i in range(0, Ne):
    file_name = f'crop{i}.jpg'
    file_path = os.path.join(folder_path, file_name)
    output = Postprocess(file_path)
    aa = output.flatten(order='C')
    # print(aa)
    # print((aa.shape))  # (16384,)
    cond_surrogate_test[:, i] = aa  # 这是一个16384行，Ne列的矩阵，每一列代表一个realization

file_path3 = r'E:\Coding_path\DiffuseVAE\scripts\Surrogate_model\testing\input_testing\cond_surrogate_test.mat'
data3 = {"cond_surrogate_test": cond_surrogate_test}  # mat文件的变量名和文件名都是cond_200.mat
scipy.io.savemat(file_path3, data3)