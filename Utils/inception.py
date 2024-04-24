import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# with open("/home/rzhou/Projects/timegan-pytorch/output/rounD_09-23_seq250_nfea10_Epoch5000+/fake_data.pickle", "rb") as fb:
#     fake_data_norm = pickle.load(fb)

file_path = '/home/rzhou/Projects/Diffusion-TS/OUTPUT/rounD_map09-23_interval10_seq500_nfea10_pad-300/ddpm_fake_rounD_map09-23_interval10_seq500_nfea10_pad-300.npy'
fake_data_norm = np.load(file_path)

df = pd.read_csv("/DATA1/rzhou/ika/multi_testcases/rounD/ori/seq500/09-23/int10/rounD_map09-23_interval10_seq500_nfea10_pad0.csv", header=0)
data = df.values
# 定义一个极端的填充值
extreme_value = -300
# 替换所有为0的值
data[data == 0] = extreme_value

scaler = MinMaxScaler()
scaler = scaler.fit(data[:,1:])
seq_length=500#要改
num_feature=10#要改
fake_data=scaler.inverse_transform(fake_data_norm.reshape(-1, num_feature)).reshape(-1, seq_length, num_feature)

fake_data[fake_data < -200] = 0

fake_data_tensor = torch.tensor(fake_data, dtype=torch.float32).to(device)

# 参数设定
input_dim = fake_data_tensor.shape[2]
hidden_dim = 50  # 减少隐藏层维度
layer_dim = 1
output_dim = 10
batch_size = 128  # 修改批次大小

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model.to(device)

# 初始化 Inception Score 累加器
total_inception_score = 0.0
num_batches = int(np.ceil(fake_data_tensor.size(0) / batch_size))

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, fake_data_tensor.size(0))
    batch_data = fake_data_tensor[start_idx:end_idx]
    
    predictions = model(batch_data)
    predictions = nn.functional.softmax(predictions, dim=1)

    # 定义 Inception Score 计算函数
    def calculate_inception_score(preds, eps=1e-16):
        pyx = preds
        py = torch.mean(pyx, 0)
        kl_div = pyx * (torch.log(pyx + eps) - torch.log(py + eps))
        kl_div = torch.sum(kl_div, 1)
        score = torch.exp(torch.mean(kl_div))
        return score

    # 计算 Inception Score
    score = calculate_inception_score(predictions)
    print(f"Inception Score for batch {batch_idx}: {score.item()}")
    total_inception_score += score.item()
    
# 计算平均 Inception Score
avg_inception_score = total_inception_score / num_batches
print("Average Inception Score:", avg_inception_score)
