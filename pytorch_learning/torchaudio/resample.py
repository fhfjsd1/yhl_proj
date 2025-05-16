# %%
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

import math
import timeit
import librosa
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import resampy
from IPython.display import Audio

DEFAULT_OFFSET = 201

# 取消DataFrame最大行列显示限制，便于查看完整数据
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

# 获取一个随时间（step）指数增长的数组，数列的值为频率（0到奈奎斯特频率），时间上是1秒
def _get_log_freq(sample_rate,max_sweep_rate,offset):
    start,stop = math.log(offset),math.log(offset+max_sweep_rate//2)
    return torch.exp(torch.linspace(start,stop,sample_rate,dtype=torch.double))-offset

# 上面的逆运算，返回上面函数返回的数组中某个频率值对应的step索引（第几个sample点）
def _get_inverse_log_freq(freq,sample_rate,offset):
    half = sample_rate//2
    return sample_rate*(math.log(1+freq/offset)/math.log(1+half/offset))

# 返回x坐标轴数据，分别是上面生成的数组中的特定频率和这些特定频率对应的时间
def _get_freq_ticks(sample_rate,offset,f_max):
    times,freq = [],[]
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10**exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
                times.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
    times.append(t_max)
    freq.append(f_max)
    return times, freq

# 生成前面的数组，计算每个微小时间内的相位增量wt，累加得到每个时间点的相位值，生成sin函数
def get_sine_sweep(sample_rate,offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal

def plot_sweep(
    waveform,
    sample_rate,
    title,
    max_sweep_rate=48000,
    offset=DEFAULT_OFFSET,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)

# 生成原始信号
sample_rate = 48000
waveform = get_sine_sweep(sample_rate)
plot_sweep(waveform, sample_rate, title="Original Waveform")

# 降采样
resample_rate = 32000
resampler = torchaudio.transforms.Resample(sample_rate,resample_rate,dtype=waveform.dtype,lowpass_filter_width=100)
resampled_waveform = resampler(waveform)
plot_sweep(resampled_waveform,resample_rate,title="Resampled Waveform")

# torchaudio.transforms.Resample的参数：lowpass_filter_width, rolloff, resampling_method
# 前两个决定理想带线内插函数的过零率（越大越接近理想化）和截止频率，用于减少频谱混叠（损失部分高频）
# 第三个是选择窗函数（和内插函数）类型，默认使用汉宁窗+sinc，用来截断内插函数，减弱频谱泄露