import torch
import numpy as np
import torch.nn as nn
from scipy.linalg import sqrtm
import pickle


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 假设 real_data_tensor 和 fake_data_tensor 已经是加载并预处理好的真实和生成的数据
# 真实数据和生成数据的张量形状应该相同，例如都是 [n_samples, seq_length, n_features]

# 加载数据示例
with open("/home/rzhou/Projects/timegan-pytorch/output/rounD_09-23_seq250_nfea10_Epoch5000+/train_data.pickle", "rb") as fb:
    real_data = pickle.load(fb)
with open("/home/rzhou/Projects/timegan-pytorch/output/rounD_09-23_seq250_nfea10_Epoch5000+/fake_data.pickle", "rb") as fb:
    fake_data = pickle.load(fb)
    
# real_data = np.load('/home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map07-17_interval10_seq500_nfea10_pad-300/samples/inD_map07-17_interval10_seq500_nfea10_pad-300_norm_truth_500_train.npy')  # 真实数据文件路径
# fake_data = np.load('/home/rzhou/Projects/Diffusion-TS/OUTPUT/inD_map07-17_interval10_seq500_nfea10_pad-300/ddpm_fake_inD_map07-17_interval10_seq500_nfea10_pad-300.npy')  # 生成数据文件路径

real_data_tensor = torch.tensor(real_data, dtype=torch.float32).to(device)
fake_data_tensor = torch.tensor(fake_data, dtype=torch.float32).to(device)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取 LSTM 最后一个时间步的输出
        return out

# 初始化特征提取器
input_dim = real_data_tensor.shape[2]  # 特征维度
hidden_dim = 50
layer_dim = 1
output_dim = 64  # 特征输出维度

extractor = FeatureExtractor(input_dim, hidden_dim, layer_dim, output_dim).to(device)
extractor.to(device)

# 提取特征
real_features = extractor(real_data_tensor).detach().cpu().numpy()
fake_features = extractor(fake_data_tensor).detach().cpu().numpy()

# 计算两组特征的均值和协方差
mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

# 计算 FID
diff = mu_real - mu_fake
covmean = sqrtm(sigma_real.dot(sigma_fake))
if np.iscomplexobj(covmean):
    covmean = covmean.real

fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
print("FID score:", fid)
