# Time : 2023/10/5 2:04
# Tong ji Marcus
# FileName: plot_zlatent_histogram.py
import scipy.io
import matplotlib.pyplot as plt

# 加载MATLAB文件
mat_data = scipy.io.loadmat(r'E:\Coding_path\DiffuseVAE\scripts\z_latent_64.mat')

# 提取数据（假设MATLAB文件中的数据存储在名为'z_latent'的变量中）
data = mat_data['z_latent']

# 转换数据为一维数组
data = data.flatten()

# 画直方图，将颜色改为灰色
plt.hist(data, bins=20, color='gray')  # 20个直方柱子，可以根据需要调整
# 关闭xy轴上的数字标签
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

plt.show()

