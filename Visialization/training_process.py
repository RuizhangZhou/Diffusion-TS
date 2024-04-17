import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map09-23_interval10_seq500_nfea10_pad-300/logs/rounD_map09-23_interval10_seq500_nfea10_pad-300.log'

# 读取日志文件
with open(log_file_path, 'r') as file:
    log_content = file.read()

# 使用正则表达式匹配每个epoch的损失值和对应的iteration
pattern = r'loss: (\d+\.\d+):.*?\|\s+(\d+)/30000'
matches = re.findall(pattern, log_content)

# 转换匹配的损失值为浮点数并解析对应的iteration
loss_values = [(float(match[0]), int(match[1])) for match in matches]

# 去除重复的损失值(通常发生在epoch的边界)，使用iteration确保正确性
unique_loss_values = []
seen = set()
for value, iteration in loss_values:
    if iteration not in seen and iteration <= 5000:  # 只考虑前5000个Epoch
        unique_loss_values.append(value)
        seen.add(iteration)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(unique_loss_values, marker='o', linestyle='-', color='b', linewidth=0.25)  # 设置更细的线条
plt.title('Training Loss Per Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.grid(True)


# 保存图像
plt.savefig('/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map09-23_interval10_seq500_nfea10_pad-300/fig/training_loss_curve.png', dpi=600)  # 保存为高分辨率图片
plt.show()
