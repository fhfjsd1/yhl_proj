import pandas as pd
import torch
import torchaudio
import numpy as np

def add_noise_with_snr(positive_sample, noise_sample, snr):
    """
    自定义加噪函数：将噪声音频按信噪比叠加到正类样本上。

    参数:
    positive_sample (Tensor): 正类样本音频数据。
    noise_sample (Tensor): 噪声音频数据。
    snr (float): 信噪比。

    返回:
    Tensor: 叠加后的音频数据。
    """
    # 确保噪声音频长度与正类样本相同
    if noise_sample.shape[1] > positive_sample.shape[1]:
        noise_sample = noise_sample[:, :positive_sample.shape[1]]
    else:
        noise_sample = torch.nn.functional.pad(noise_sample, (0, positive_sample.shape[1] - noise_sample.shape[1]))

    # 计算信噪比
    positive_power = positive_sample.norm(p=2)
    noise_power = noise_sample.norm(p=2)
    scale = positive_power / (10 ** (snr / 20) * noise_power)

    # 叠加噪声音频
    augmented_sample = positive_sample + scale * noise_sample
    return augmented_sample

# # 读取 CSV 文件
# csv_file = self.hparams.noise_annotation
# df = pd.read_csv(csv_file)
# # 获取第三列的所有噪声地址
# noise_paths = df.iloc[:, 2].tolist()

# for i in range(self.hparams.batch_size):
# # 在训练过程中应用数据增强
#     noise_sample_path = random.choice(noise_paths)

#     # 加载音频文件
#     noise_sample, _ = torchaudio.load(noise_sample_path)
#     noise_sample = noise_sample.to(self.device)

#     # 应用数据增强
#     snr = 0  # 设置信噪比
#     wavs[i] = add_noise_with_snr(wavs[i].unsqueeze(0), noise_sample, snr)