import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_prepare_data(norm_path, csv_path):
    # Load normalized data
    fake_data_norm = np.load(norm_path)

    # Load original data for scaling purposes
    df = pd.read_csv(csv_path, header=0)
    data = df.values
    extreme_value = -300
    data[data == 0] = extreme_value
    scaler = MinMaxScaler().fit(data[:, 1:])

    # Restore fake_data from normalized state to original scale
    num_feature = df.shape[1] - 1
    seq_length = 250
    fake_data = scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)
    return fake_data

def calculate_velocity_and_heading(data, delta_time=0.04):
    velocities = np.zeros((data.shape[0]-1, 2))
    headings = np.zeros(data.shape[0]-1)
    for i in range(1, data.shape[0]):
        dx = data[i, 0] - data[i-1, 0]
        dy = data[i, 1] - data[i-1, 1]
        velocities[i-1] = [dx / delta_time, dy / delta_time]
        headings[i-1] = np.arctan2(dy, dx)
    headings = np.unwrap(headings)
    return velocities, headings

def kinematic_bicycle_model(velocities, headings, wheelbase=2.5, delta_time=0.04, min_velocity=0.1):
    steer_angles = np.zeros(velocities.shape[0])
    previous_heading = headings[0]
    for i in range(1, velocities.shape[0]):
        v = np.linalg.norm(velocities[i])
        if v < min_velocity:
            v = min_velocity
        current_heading = headings[i]
        omega = (current_heading - previous_heading) / delta_time
        steer = np.arctan((wheelbase * omega) / v)
        steer_angles[i] = np.degrees(steer)
    return steer_angles

def process_datasets(data_pairs):
    log_file = "250_steer_analysis_log.txt"
    with open(log_file, 'w') as log:
        for norm_path, csv_path in data_pairs:
            fake_data = load_and_prepare_data(norm_path, csv_path)
            all_steer_angles = []
            for case_index in range(fake_data.shape[0]):
                for vehicle_index in range(0, fake_data.shape[2], 2):
                    data = fake_data[case_index, :, vehicle_index:vehicle_index+2]
                    velocities, headings = calculate_velocity_and_heading(data)
                    steer_angles = kinematic_bicycle_model(velocities, headings)
                    all_steer_angles.append(steer_angles)
            all_steer_angles_flat = np.concatenate(all_steer_angles)
            mean_absolute_steer = np.mean(np.abs(all_steer_angles_flat))
            log.write(f"Results for {norm_path} with {csv_path}:\n")
            log.write(f"Mean absolute steering angle: {mean_absolute_steer} degrees\n\n")

if __name__ == "__main__":
    data_pairs = [
        ('/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_multi_09-23_interval250_numfea10_nopad/ddpm_fake_rounD_multi_09-23_interval250_numfea10_nopad.npy', '/DATA1/rzhou/ika/multi_testcases/rounD/ori/nopad/rounD_multi_09-23_interval250_numfea10_nopad.csv'),
        ('/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_multi_09-23_interval250_numfea10_nopad/samples/rounD_multi_09-23_interval250_numfea10_nopad_norm_truth_250_train.npy', '/DATA1/rzhou/ika/multi_testcases/rounD/ori/nopad/rounD_multi_09-23_interval250_numfea10_nopad.csv'),
        ('/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_single_09-23_seq250/ddpm_fake_rounD_single_09-23_seq250.npy', '/DATA1/rzhou/ika/single_testcases/rounD/rounD_single_09-23_seq250.csv'),
        ('/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_single_09-23_seq250/samples/rounD_single_09-23_seq250_norm_truth_250_train.npy', '/DATA1/rzhou/ika/single_testcases/rounD/rounD_single_09-23_seq250.csv'),
        # Add more pairs as needed
    ]
    process_datasets(data_pairs)
