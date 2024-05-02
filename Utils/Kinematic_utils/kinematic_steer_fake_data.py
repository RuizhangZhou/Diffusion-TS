import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载和准备数据
file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map18-29_interval1_seq500_nfea10_pad-300/ddpm_fake_inD_map18-29_interval1_seq500_nfea10_pad-300.npy'
fake_data_norm = np.load(file_path)

# 假设原始数据路径和原始数据的处理
df = pd.read_csv("/DATA1/rzhou/ika/multi_testcases/inD/inD_map18-29_interval1_seq500_nfea10.csv", header=0)
data = df.values
# 定义一个极端的填充值
extreme_value = -300
# 替换所有为0的值
data[data == 0] = extreme_value
scaler = MinMaxScaler().fit(data[:, 1:])

seq_length = 500
num_feature = df.shape[1] - 1
# 将fake_data从归一化状态恢复到原始尺度
fake_data = scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)

# 处理每个测试用例
delta_time = 0.04  # 假设的每帧间隔时间，单位秒
wheelbase = 2.5  # 假设的车辆轴距，单位为米

def calculate_steering_angles(data):
    steer = np.zeros(data.shape[0])
    for i in range(1, data.shape[0]):
        dx = data[i, 0] - data[i - 1, 0]
        dy = data[i, 1] - data[i - 1, 1]
        velocity = np.sqrt(dx**2 + dy**2) / delta_time
        if i > 1:
            previous_heading = np.arctan2(data[i-1, 1] - data[i-2, 1], data[i-1, 0] - data[i-2, 0])
            current_heading = np.arctan2(dy, dx)
            omega = (current_heading - previous_heading) / delta_time
            steer[i] = np.arctan((wheelbase * omega) / velocity)
    return steer

steering_angles = []
for case_index in range(fake_data.shape[0]):
    case_data = fake_data[case_index]
    steer = calculate_steering_angles(case_data[:, :2])
    steering_angles.append(steer)

# 将所有结果转换为一个DataFrame并保存到CSV文件
steer_df = pd.DataFrame(steering_angles).transpose()  # 转置以使每列代表一个测试用例
output_path = '/home/rzhou/Projects/Diffusion-TS/Utils/Kinematic_utils/steer/steer_output_steering_angles.csv'
steer_df.to_csv(output_path, index=False)
print(f"Steering data has been saved to {output_path}")
