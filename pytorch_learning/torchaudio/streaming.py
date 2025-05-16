# %%
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

import matplotlib.pyplot as plt
from torchaudio.io import StreamReader
import IPython

import requests
import sys

base_url = "https://download.pytorch.org/torchaudio/tutorial-assets"
AUDIO_URL = f"{base_url}/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
VIDEO_URL = f"{base_url}/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"

# print("当前Python解释器的模块和包搜索路径：\n", sys.path)
# print(torchaudio.utils.ffmpeg_utils.get_input_devices())

# 因为这里IO一个filelike的对象需要用到read（必须）和seek方法（如果有的话）（默认是使用 io.RawIOBase.XXX），
# 为了避免第三方库也有seek方法，所以建一个类藏起来seek方法
def _mask_seek(obj):
    class _wrapper:
        def __init__(self,obj) -> None:
            self.obj = obj
            
        def read(self,n):
            return self.obj.read(n)
    
    return _wrapper(obj)

# response = requests.get(AUDIO_URL,stream=True)
# s = StreamReader(_mask_seek(response.raw))

# print(s.num_src_streams)
# print(s.get_src_stream_info(0))

# 大坑：首先整个stream都是基于ffmpeg这个多媒体编辑器的，然后教程中使用conda安装的库为了保证最大兼容性，
# 打包时不会启用某些编译配置选项（比如--enable-alsa等等），就会导致无法使用部分多媒体设备，需要重新寻找新的版本，
# 或者自己编译安装（更麻烦）。然后对于各种媒体设备，都有一个类似驱动的东西，作为程序访问硬件数据的接口（alsa,oss,v4f1。。。）
# 也就是下面的format，下面的参数传递规则需要参考ffmpeg的官方文档
# streamer = StreamReader(
#     src="hw:2,0",
#     format="alsa",
#     option={"sample_rate":"48000","channels":"2"}
#     )

streamer = StreamReader(
    src="sine=sample_rate=8000:frequency=6",
    format="lavfi")

# streamer = torchaudio.io.StreamReader(VIDEO_URL)
for i in range(streamer.num_src_streams):
    print(streamer.get_src_stream_info(i))
    
streamer.add_basic_audio_stream(frames_per_chunk=8000,
                                sample_rate=8000)

for i in range(streamer.num_out_streams):
    print(streamer.get_out_stream_info(i))
    
fig,ax = plt.subplots(2,1)
[waveform] = next(streamer.stream())
time_axes = torch.arange(0,8000)/8000
ax[0].specgram(waveform[:,0],Fs = 8000)
ax[1].plot(time_axes,waveform[:,0],linewidth = 1)


# fmt: off
descs = [
    # No filtering
    "anull",
    # Apply a highpass filter then a lowpass filter
    "highpass=f=200,lowpass=f=1000",
    # Manipulate spectrogram
    (
        "afftfilt="
        "real='hypot(re,im)*sin(0)':"
        "imag='hypot(re,im)*cos(0)':"
        "win_size=512:"
        "overlap=0.75"
    ),
    # Manipulate spectrogram
    (
        "afftfilt="
        "real='hypot(re,im)*cos((random(0)*2-1)*2*3.14)':"
        "imag='hypot(re,im)*sin((random(1)*2-1)*2*3.14)':"
        "win_size=128:"
        "overlap=0.8"
    ),
]
# fmt: on
print("-----"*5)
sample_rate = 8000
streamer2 = StreamReader(AUDIO_URL)

for desc in descs:
    streamer2.add_audio_stream(frames_per_chunk=40000,
                              filter_desc=f"aresample={sample_rate},{desc},aformat=sample_fmts=fltp")

chunks = next(streamer2.stream())

def _display(n):
    for i in range(streamer2.num_src_streams):
        print(streamer2.get_src_stream_info(i))
    for i in range(streamer2.num_out_streams):
        print(streamer2.get_out_stream_info(i))
    
    fig,axes = plt.subplots(2,1)
    waveform = chunks[n][:,0]
    axes[0].plot(waveform)
    axes[0].grid(True)
    axes[0].set_ylim([-1,1])
    axes[1].specgram(waveform,Fs=sample_rate)
    fig.tight_layout()
    
for i in range(4):
    _display(i)
    
plt.show()

streamer = StreamReader(VIDEO_URL)
for i in range(streamer.num_src_streams):
    print(streamer.get_src_stream_info(i))

streamer.add_basic_audio_stream(frames_per_chunk = 8000,
                                sample_rate = 8000)
streamer.add_basic_audio_stream(frames_per_chunk = 16000,
                                sample_rate = 16000)
streamer.add_basic_video_stream(frames_per_chunk=1,
                                 frame_rate =1,
                                 width = 960,
                                 height = 540,
                                 format = "rgb24")
streamer.add_basic_video_stream(frames_per_chunk=3,
                                frame_rate=3,
                                width=320,
                                height=320,
                                format="gray")
streamer.remove_stream(1)

for i in range(streamer.num_out_streams):
    print(streamer.get_out_stream_info(i))

streamer.seek(10.0)
n_iteratons = 3
waveforms,vids1,vids2 = [],[],[]
for i,(waveform,vid1,vid2) in enumerate(streamer.stream(),start=1):
    waveforms.append(waveform)
    vids1.append(vid1)
    vids2.append(vid2)
    if i == n_iteratons:
        break
    
k = 3
fig = plt.figure()
gs = fig.add_gridspec(3, k * n_iteratons)
for i, waveform in enumerate(waveforms):
    ax = fig.add_subplot(gs[0, k * i : k * (i + 1)])
    ax.specgram(waveform[:, 0], Fs=8000)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"Iteration {i}")
    if i == 0:
        ax.set_ylabel("Stream 0")
for i, vid in enumerate(vids1):
    ax = fig.add_subplot(gs[1, k * i : k * (i + 1)])
    ax.imshow(vid[0].permute(1, 2, 0))  # NCHW->HWC
    ax.set_yticks([])
    ax.set_xticks([])
    if i == 0:
        ax.set_ylabel("Stream 1")
for i, vid in enumerate(vids2):
    for j in range(3):
        ax = fig.add_subplot(gs[2, k * i + j : k * i + j + 1])
        ax.imshow(vid[j].permute(1, 2, 0), cmap="gray")
        ax.set_yticks([])
        ax.set_xticks([])
        if i == 0 and j == 0:
            ax.set_ylabel("Stream 2")
plt.tight_layout()
plt.show()
