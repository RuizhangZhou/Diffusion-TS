import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 加载和准备数据
file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_single_09-23_seq250/ddpm_fake_rounD_single_09-23_seq250.npy'
fake_data_norm = np.load(file_path)

# 假设原始数据路径和原始数据的处理
df = pd.read_csv("/DATA1/rzhou/ika/single_testcases/rounD/rounD_single_09-23_seq250.csv", header=0)
data = df.values
scaler = MinMaxScaler().fit(data[:, 1:])

seq_length = 250
num_feature = df.shape[1] - 1
# 将fake_data从归一化状态恢复到原始尺度
fake_data = scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)

# 设置极端值并替换
#extreme_value = -300
fake_data[fake_data < -200] = 0

def calculate_euclidean_loss(df):
    df['x_pred'] = df['x_dot_theory']
    df['y_pred'] = df['y_dot_theory']
    df['x_actual'] = df['xVelocity']
    df['y_actual'] = df['yVelocity']
    df['euclidean_loss'] = np.sqrt((df['x_pred'] - df['x_actual'])**2 + (df['y_pred'] - df['y_actual'])**2)
    return df['euclidean_loss'].mean()

def calculate_model_differences(df, delta_degrees=5):
    theta_rad = np.radians(df['heading'])
    v = np.sqrt(df['xVelocity']**2 + df['yVelocity']**2)
    delta = np.radians(delta_degrees)
    beta = np.arctan(np.tan(delta) * 0.5)
    df['x_dot_theory'] = v * np.cos(theta_rad + beta)
    df['y_dot_theory'] = v * np.sin(theta_rad + beta)
    return df

# 处理每个测试用例
delta_time = 0.04  # 假设的每帧间隔时间，单位秒

def calculate_velocity_and_heading(data):
    # 数据列假设是连续的[x, y]坐标
    velocities = np.zeros_like(data)
    headings = np.zeros(data.shape[0])
    
    for i in range(1, data.shape[0]):
        dx = data[i, 0] - data[i - 1, 0]
        dy = data[i, 1] - data[i - 1, 1]
        velocities[i, 0] = dx / delta_time
        velocities[i, 1] = dy / delta_time
        angle_rad = np.arctan2(dy, dx)
        calculated_heading = np.degrees(angle_rad)
        headings[i] = (90 - calculated_heading) % 360

    return velocities, headings

def find_first_and_last_non_zero(x_column, y_column):
    non_zero_mask = (x_column != 0) & (y_column != 0)
    if not np.any(non_zero_mask):
        return None, None
    initial_frame = np.argmax(non_zero_mask)
    final_frame = non_zero_mask.size - 1 - np.argmax(non_zero_mask[::-1])
    return initial_frame, final_frame

total_loss = 0
num_cases = 0
print(fake_data.shape[0])
# 计算每个测试用例的损失
for case_index in range(fake_data.shape[0]):
    #print(case_index)
    case_data = fake_data[case_index]
    velocities, headings = calculate_velocity_and_heading(case_data[:, :2])
    
    # 合并 x, y, xVelocity, yVelocity, heading 到 DataFrame
    test_case = np.hstack((case_data[:, :2], velocities, headings.reshape(-1, 1)))
    df = pd.DataFrame(test_case, columns=['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'heading'])

    # 标记和找到有效数据的开始和结束
    initial_frame, final_frame = find_first_and_last_non_zero(df['xCenter'], df['yCenter'])
    if initial_frame is None or final_frame is None:
        continue  # 如果全是极端值，跳过这个测试用例

    # 裁剪到有效数据
    df = df.iloc[initial_frame:final_frame+1]

    # 计算模型差异和损失
    df = calculate_model_differences(df)
    loss = calculate_euclidean_loss(df)
    total_loss += loss
    num_cases += 1

# 计算平均损失
average_loss = total_loss / num_cases if num_cases > 0 else None

print(f"Average Kinematic Bicycle Model Loss: {average_loss}")
