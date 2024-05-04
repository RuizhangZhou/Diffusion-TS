import numpy as np
import pandas as pd

def calculate_steering_angle(df, wheelbase=2.5, delta_time=0.04):
    # 计算车速
    df['velocity'] = np.sqrt(df['xVelocity']**2 + df['yVelocity']**2)
    # 处理航向角数据，使用np.unwrap防止角度跳变
    df['heading_rad'] = np.radians(df['heading'])
    df['heading_rad'] = np.unwrap(df['heading_rad'])
    # 计算角速度
    df['omega'] = np.gradient(df['heading_rad']) / delta_time
    # 计算转向角
    df['steer'] = np.arctan2((wheelbase * df['omega']), df['velocity'])
    df['steer_degrees'] = np.degrees(df['steer'])
    # 调整角度范围，正负角度处理
    df['steer_degrees'] = (df['steer_degrees'] + 180) % 360 - 180
    return df[['steer_degrees']]  # 返回只有转向角度的DataFrame

def load_and_process_data(file_path, wheelbase):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 确保数据列是正确的数据类型
    df['xVelocity'] = df['xVelocity'].astype(float)
    df['yVelocity'] = df['yVelocity'].astype(float)
    df['heading'] = df['heading'].astype(float)
    # 处理数据并计算转向角
    return calculate_steering_angle(df, wheelbase, delta_time=0.04)  # 调整delta_time为您的数据实际时间间隔

# 文件路径和车辆轴距
file_path = '/DATA1/rzhou/ika/rounD/data/00_tracks.csv'
wheelbase = 2.5

# 加载数据并计算转向角
steer_degrees = load_and_process_data(file_path, wheelbase)

# 将结果保存到CSV文件
output_path = '/home/rzhou/Projects/Diffusion-TS/Utils/Kinematic_utils/steer/steer_output_steering_angles.csv'
steer_degrees.to_csv(output_path, index=False, header=True)
print(f"Steering angle data in degrees has been saved to {output_path}")
