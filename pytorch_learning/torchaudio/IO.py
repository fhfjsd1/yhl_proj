# %%
import torch
import torchaudio

import sys

# import io
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu'

# import tarfile
# import tempfile

import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

print("当前Python解释器的版本：", sys.version)
print("当前Python解释器的模块和包搜索路径：\n", sys.path)
print(torch.__version__)
print(torchaudio.__version__)



SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)
SAMPLE_WAV_8000 = download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav"
)
print(SAMPLE_GSM)

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj) -> None:
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)

metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)

waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels ==1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis,waveform[c],linewidth = 1)
        axes[c].grid(True)
        axes[c].set_ylabel(f"Channel {c+1}")
    
    figure.suptitle("waveform")
    #plt.show()

plot_waveform(waveform,sample_rate)

def plot_specgram(waveform,sample_rate):
    waveform.numpy()
    
    num_channels,num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels ==1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c],Fs=sample_rate)
        axes[c].set_ylabel(f"Channel {c+1}")
    
    figure.suptitle("specgram")
    #plt.show()
    
plot_specgram(waveform,sample_rate)

frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds

url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
print("Fetching all the data...")
with requests.get(url, stream=True) as response:
    waveform1, sample_rate1 = torchaudio.load(_hide_seek(response.raw))
    waveform1 = waveform1[:, frame_offset : frame_offset + num_frames]
    print(f" - Fetched {response.raw.tell()} bytes")

print("Fetching until the requested frames are available...")
with requests.get(url, stream=True) as response:
    waveform2, sample_rate2 = torchaudio.load(
        _hide_seek(response.raw), frame_offset=frame_offset, num_frames=num_frames
    )
    print(f" - Fetched {response.raw.tell()} bytes")

print("Checking the resulting waveform ... ", end="")
assert (waveform1 == waveform2).all()
print("matched!")
    

# %%
