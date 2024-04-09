import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import os

Dtype="rounD"
index_map=1
# fake_data_file.npy
file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map01_interval1_seq500_nfea10_pad-300/ddpm_fake_rounD_map01_interval1_seq500_nfea10_pad-300.npy'
fake_data_norm = np.load(file_path)

df = pd.read_csv("/DATA1/rzhou/ika/multi_testcases/rounD/ori/seq500/00-01/rounD_map01_interval1_seq500_nfea10.csv", header=0)
data = df.values

# 定义一个极端的填充值
extreme_value = -300
# 替换所有为0的值
data[data == 0] = extreme_value

scaler = MinMaxScaler()
scaler = scaler.fit(data[:,1:])
seq_length=500
num_feature=10
fake_data=scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)

fake_data[fake_data < -200] = 0

fake_index = 689
# 假设data是已经加载的数据
cur_fake_data=fake_data[fake_index]
n, n_features = cur_fake_data.shape

def find_first_and_last_non_zero(x_column, y_column):
    # 生成一个标记非零元素位置的布尔数组
    non_zero_mask = (x_column != 0) & (y_column != 0)
    # 如果全部是零，则返回None
    if not np.any(non_zero_mask):
        return None, None
    # 从前往后找第一个非零值的位置
    initial_frame = np.argmax(non_zero_mask)
    # 反转数组后，再次找第一个非零值的位置，实际上是原数组中最后一个非零值的位置
    final_frame = non_zero_mask.size - 1 - np.argmax(non_zero_mask[::-1])
    return initial_frame, final_frame


# 初始化两个空的DataFrame来分别存储tracks的元数据和tracks的详细信息
tracks_meta_columns = ['recordingId', 'trackId', 'initialFrame', 'finalFrame', 'numFrames', 'width', 'length', 'class']
tracks_columns = [
    'recordingId', 'trackId', 'frame', 'trackLifetime',
    'xCenter', 'yCenter', 'heading', 'width', 'length',
    'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
    'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration'
]

tracks_meta_df = pd.DataFrame(columns=tracks_meta_columns)
tracks_df = pd.DataFrame(columns=tracks_columns)

delta_time = 0.04  # 假设的每帧间隔时间，单位秒
recordingId = f'fake_{Dtype}_{index_map:02d}_{fake_index}'
frame_rate = 25  # 假设的帧率

# 遍历每个track
for track_id in range(n_features // 2):
    x_column = cur_fake_data[:, track_id * 2]
    y_column = cur_fake_data[:, track_id * 2 + 1]
    initial_frame, final_frame = find_first_and_last_non_zero(x_column, y_column)
    
    if initial_frame is None or final_frame is None:
        continue  # 如果这个track全是零，则跳过

    # 构造tracksMeta的元数据行
    track_meta = {
        'recordingId': recordingId,
        'trackId': track_id,
        'initialFrame': initial_frame,
        'finalFrame': final_frame,
        'numFrames': final_frame - initial_frame + 1,
        'width': 2,
        'length': 4,
        'class': 'car'
    }
    tracks_meta_df = pd.concat([tracks_meta_df, pd.DataFrame([track_meta])], ignore_index=True)

    # 添加到tracks_df
    for frame in range(initial_frame, final_frame + 1):
        x_velocity = (x_column[min(frame + 1, final_frame)] - x_column[frame]) / delta_time
        y_velocity = (y_column[min(frame + 1, final_frame)] - y_column[frame]) / delta_time

        track_data = {
            'recordingId': recordingId,
            'trackId': track_id,
            'frame': frame,
            'trackLifetime': frame - initial_frame,
            'xCenter': x_column[frame],
            'yCenter': y_column[frame],
            'heading': 100,
            'width': 2,
            'length': 4,
            'xVelocity': x_velocity,
            'yVelocity': y_velocity,
            'xAcceleration': 0,
            'yAcceleration': 0,
            'lonVelocity': 0,
            'latVelocity': 0,
            'lonAcceleration': 0,
            'latAcceleration': 0
        }
        tracks_df = pd.concat([tracks_df, pd.DataFrame([track_data])], ignore_index=True)

def checkDirExistOrCreate(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# 定义文件路径
tracks_meta_csv_dir = f'/home/rzhou/Projects/Diffusion-TS/Data/ika_format_csv/csv/fake_{Dtype}_{index_map:02d}'
checkDirExistOrCreate(tracks_meta_csv_dir)
tracks_meta_csv_path = f'{tracks_meta_csv_dir}/fake_{Dtype}_{index_map:02d}_{fake_index}_tracksMeta.csv'
tracks_csv_path = f'{tracks_meta_csv_dir}/fake_{Dtype}_{index_map:02d}_{fake_index}_tracks.csv'

# 清空文件内容
open(tracks_meta_csv_path, 'w').close()
open(tracks_csv_path, 'w').close()

# 现在文件已被清空，可以安全地写入新内容
tracks_meta_df.to_csv(tracks_meta_csv_path, index=False)
tracks_df.to_csv(tracks_csv_path, index=False)

