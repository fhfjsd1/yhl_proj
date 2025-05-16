# %%

# 初始化模块，首先运行这个单元格
import torch
import torchaudio
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt

SAMPLE_WAV = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav")
SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")

assets = [SAMPLE_WAV, SAMPLE_RIR, SAMPLE_SPEECH, SAMPLE_NOISE]
for sample in assets:
    print(torchaudio.info(sample))


# 绘图函数
def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.numpy()
    if waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    fig, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c])
        axes[c].grid(True)
        axes[c].set_ylabel(f"Channel{c+1}")
    fig.suptitle(title)


def plot_specgram(waveform, sample_rate, title="Specgram"):
    waveform = waveform.numpy()
    if waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T
    num_channels, _ = waveform.shape

    fig, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        axes[c].set_ylabel(f"Channel{c+1}")
    fig.suptitle(title)

# %%

# %%
waveform1, sample_rate = torchaudio.load(SAMPLE_WAV, channels_first=False)

effect = ",".join(
    [
        "lowpass=frequency=300:poles=1",
        "atempo=0.8",
        "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3",
    ],
)

def apply_effect(waveform, sample_rate, effect):
    effector = torchaudio.io.AudioEffector(effect)
    return effector.apply(waveform, sample_rate)

waveform2 = apply_effect(waveform1, sample_rate, effect)

# 第一段音频的加特效实验
plot_waveform(waveform1, sample_rate, title="Original")
plot_specgram(waveform1, sample_rate, title="Original")

plot_waveform(waveform2, sample_rate, title="effects_applied")
plot_specgram(waveform2, sample_rate, title="effects_applied")

# %%

# %%
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)")
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")

# 清洗数据，只提取RIR的主脉冲部分，然后归一化
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)
plot_waveform(rir, sample_rate, "Room Impulse Response(normalized)")

speech, _ = torchaudio.load(SAMPLE_SPEECH)
augmented_speech = torchaudio.functional.fftconvolve(speech, rir)

plot_waveform(speech, sample_rate, title="Original")
plot_specgram(speech, sample_rate, title="Original")

plot_waveform(augmented_speech, sample_rate, title="RIR Applied")
plot_specgram(augmented_speech, sample_rate, title="RIR Applied")
plt.show()

# %%

# %%
# 加噪声（不同信噪比dB）
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, 0 : speech.shape[1]]
plot_waveform(noise, sample_rate, title="noise")
plot_specgram(noise, sample_rate, title="noise")
snr_dbs = torch.tensor([20, 10, 3])
noisy_speeches = torchaudio.functional.add_noise(speech, noise, snr_dbs)

# %%
waveform, sample_rate = torchaudio.load(SAMPLE_SPEECH, channels_first=False)

def apply_codec(waveform,sample_rate,format,encoder=None):
    encoder = torchaudio.io.AudioEffector(format=format,encoder=encoder)
    return encoder.apply(waveform,sample_rate)

mulaw = apply_codec(waveform, sample_rate, "wav", encoder="pcm_mulaw")
plot_waveform(mulaw, sample_rate, title="8 bit mu-law")
plot_specgram(mulaw, sample_rate, title="8 bit mu-law")
# %%

speech_raw,sample_rate = torchaudio.load(SAMPLE_SPEECH)

noise,_=torchaudio.load(SAMPLE_NOISE)
noise = noise[:,0:speech_raw.shape[1]]
snr = torch.tensor([8])
speech_noisy = torchaudio.functional.add_noise(speech_raw,noise,snr)

rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.linalg.vector_norm(rir, ord=2)
augmented_speech = torchaudio.functional.fftconvolve(speech_noisy,rir)

effect = ",".join(
    [
        "lowpass=frequency=4000:poles=1",
        "compand=attacks=0.02:decays=0.05:points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:gain=-8:volume=-7:delay=0.05",
    ]
)
effector = torchaudio.io.AudioEffector(effect,format="g722")
output = effector.apply(augmented_speech.T,8000)

plot_waveform(output,8000)
plot_specgram(output,8000)
# %%
