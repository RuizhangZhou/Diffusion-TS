import numpy as np
import pandas as pd
import glob  # 用于文件路径的匹配
import os

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

def process_file(file_path):
    # 如果知道引起问题的列的数据类型，可以显式指定。假设列32是整数，列35是浮点数。
    dtype_spec = {32: int, 35: float}
    try:
        tracks_df = pd.read_csv(file_path, dtype=dtype_spec)
    except ValueError:
        # 如果dtype_spec不匹配或者有问题，可以尝试不指定dtype
        tracks_df = pd.read_csv(file_path, low_memory=False)
    tracks_df['xCenter'] = tracks_df['xCenter'].astype(float)
    tracks_df['yCenter'] = tracks_df['yCenter'].astype(float)
    tracks_df['xVelocity'] = tracks_df['xVelocity'].astype(float)
    tracks_df['yVelocity'] = tracks_df['yVelocity'].astype(float)
    tracks_df['heading'] = tracks_df['heading'].astype(float)
    tracks_df = calculate_model_differences(tracks_df)
    loss = calculate_euclidean_loss(tracks_df)
    return loss, len(tracks_df)

# 定义文件夹路径
dataset_type='exiD'
folder_path = f'/DATA1/rzhou/ika/{dataset_type}/data/'

# 指定文件号范围
idx_start=78
idx_end=92
file_numbers = range(idx_start, idx_end+1)  # 修改您的范围

# 构造文件路径列表
file_paths = [f"{folder_path}{str(i).zfill(2)}_tracks.csv" for i in file_numbers]

# 计算每个文件的损失并存储，同时获取行数
losses = []
total_rows = 0
for fp in file_paths:
    if os.path.exists(fp):
        loss, rows = process_file(fp)
        losses.append((loss, rows))
        total_rows += rows

# 计算加权平均损失
if losses:
    weighted_loss = sum(loss * rows for loss, rows in losses) / total_rows
else:
    weighted_loss = None

print(f"Weighted Average Kinematic Bicycle Model Loss {dataset_type} {idx_start:02d} to {idx_end:02d}: {weighted_loss}")
