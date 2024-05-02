import numpy as np
import pandas as pd

def calculate_steering_angle(df, wheelbase=2.5):
    # 计算车速
    df['velocity'] = np.sqrt(df['xVelocity']**2 + df['yVelocity']**2)
    # 计算角速度
    df['omega'] = np.gradient(np.radians(df['heading']))
    # 计算转向角
    df['steer'] = np.arctan2((wheelbase * df['omega']), df['velocity'])
    df['steer_degrees'] = np.degrees(df['steer'])  # 转为度数
    return df[['steer_degrees']]  # 返回只有转向角度的DataFrame

def load_and_process_data(file_path, wheelbase):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 确保数据列是正确的数据类型
    df['xVelocity'] = df['xVelocity'].astype(float)
    df['yVelocity'] = df['yVelocity'].astype(float)
    df['heading'] = df['heading'].astype(float)
    # 处理数据并计算转向角
    return calculate_steering_angle(df, wheelbase)

# 文件路径和车辆轴距
file_path = '/DATA1/rzhou/ika/rounD/data/00_tracks.csv'  # 这里替换成您的文件路径
wheelbase = 2.5  # 车辆的轴距，单位为米

# 加载数据并计算转向角
steer_df = load_and_process_data(file_path, wheelbase)

# 输出结果中的部分数据，例如转向角
print(steer_df.head())

# 将结果保存到CSV文件
output_path = '/home/rzhou/Projects/Diffusion-TS/Utils/Kinematic_utils/steer/steer_output_steering_angles.csv'  # 输出文件的路径
steer_df.to_csv(output_path, index=False)
print(f"Steering angle data has been saved to {output_path}")
