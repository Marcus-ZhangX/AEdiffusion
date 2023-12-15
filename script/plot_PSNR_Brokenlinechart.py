import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(family='Times New Roman', size=11)
# 六组数据，每组包含六个值
data = [
    [3.8568, 3.9097, 3.9152, 3.9960, 4.0868, 4.0943],
    [3.8568, 3.9066, 3.9245, 4.0052, 4.0591, 4.0611],
    [3.8470, 3.8410, 3.9742, 4.0547, 4.0824, 4.1040],
    [3.8470, 3.8538, 3.9092, 3.9842, 4.0676, 4.0907],
    [3.8423, 3.8406, 3.9873, 4.0758, 4.1199, 4.1254],
    [3.8423, 3.8230, 3.8155, 3.9788, 4.0893, 4.1240]
]

# 不同的线条样式和标记符号
line_styles = [':', ':', '--', '--', '-', '-']
marker_symbols = ['o', 'o', '^', '^', 'D', 'D']

# 不同的线条颜色
# line_colors = ['royalblue', 'firebrick', 'royalblue', 'firebrick', 'royalblue', 'firebrick']
line_colors = ['limegreen', 'green', 'royalblue', 'mediumblue', 'lightcoral', 'firebrick']
# 不同的线条粗细
linewidths = [2, 2, 2, 2, 2, 2]

# 标签名称
labels = [
    '400Ne with Surrogate',
    '400Ne without Surrogate',
    '600Ne with Surrogate',
    '600Ne without Surrogate',
    '800Ne with Surrogate',
    '800Ne without Surrogate'
]

# 绘制折线图
for i in range(6):
    plt.plot(data[i], label=labels[i], linestyle=line_styles[i], marker=marker_symbols[i], color=line_colors[i], linewidth=linewidths[i])

# 添加图例
legend = plt.legend(prop=font, frameon=True)
frame = legend.get_frame()
frame.set_linewidth(1.0)  # 设置图例边框线粗
frame.set_edgecolor('black')  # 设置图例边框颜色为黑色
# 显示折线图
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.show()
