import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import random
from PIL import Image

# fake_data_file.npy
file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map19_interval100_seq1000_reduced_nfea40/ddpm_fake_inD_map19_interval100_seq1000_reduced_nfea40.npy'

# 使用numpy的load函数读取npy文件
fake_data_norm = np.load(file_path)


df = pd.read_csv("/DATA1/rzhou/ika/multi_testcases/inD/reduced/inD_map19_interval100_seq1000_reduced_nfea40.csv", header=0)
data = df.values
scaler = MinMaxScaler()
scaler = scaler.fit(data[:,1:])
seq_length=1000
num_feature=40
fake_data=scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)

# 将所有绝对值小于10的元素替换为0
fake_data[np.abs(fake_data) < 10] = 0

# 假设 fake_data 是一个形状为 (num_cases, 1500, 60) 的数组
# 生成1个case
# random_index = random.randint(0, fake_data.shape[0] - 1)
# #random_index = 111
# print(f"Random index: {random_index}")

# 生成多个case
num_cases=5
random_indices = np.random.choice(fake_data.shape[0], num_cases, replace=False)
print(f"Random indices: {random_indices}")

num_v=20

# 载入背景图片
#bg_image_path = '/DATA1/rzhou/ika/rounD/data/02_background.png'
bg_image_path = '/DATA1/rzhou/ika/inD/data/19_background.png'
bg_img = Image.open(bg_image_path)
# 获取图像的宽度和高度
width, height = bg_img.size
# 将图像的尺寸转换为英寸（matplotlib的figsize是以英寸为单位的）
# 假设希望每100像素对应于1英寸，则可以按如下方式计算figsize：
figsize = (width / 100, height / 100)

#设置num_v个颜色
colors = plt.cm.jet(np.linspace(0, 1, num_v))


for random_index in random_indices:
    # 创建图形和轴，使用图像的原始尺寸
    fig, ax = plt.subplots(figsize=figsize)
    bg_img = plt.imread(bg_image_path)
    # inD-19
    ax.set_xlim(0, 80)
    ax.set_ylim(-60, 0)
    ax.imshow(bg_img, extent=[0, 117, -78, 0])

    # rounD-02
    # ax.set_xlim(0, 170)
    # ax.set_ylim(-95, 0)
    # ax.imshow(bg_img, extent=[0, 170, -95, 0])

    lines = [ax.plot([], [], marker='o', linestyle='', color=colors[i])[0] for i in range(num_v)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        for j, line in enumerate(lines):
            x = fake_data[random_index][i, j*2]
            y = fake_data[random_index][i, j*2+1]
            # 如果x或y为0，则不显示该点
            if x == 0 and y == 0:
                line.set_data([], [])
            else:
                line.set_data(x, y)
        return lines

    anim = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=40, blit=True)

    #plt.legend([f"Point {i+1}" for i in range(num_v)], loc='upper right', fontsize='small')
    # 在每个动画循环的末尾，但在保存动画之前，添加图例
    plt.legend([f"Point {i+1}" for i in range(num_v)], loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')


    # 保存动画为MP4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(f'/home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map19_interval100_seq1000_reduced_nfea40/animations/{random_index}.mp4', writer=writer)
    plt.close(fig)  # 关闭当前绘图窗口，防止过多图形打开
