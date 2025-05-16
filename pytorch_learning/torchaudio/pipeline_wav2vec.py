# speech recognition using using pre-trained models from wav2vec 2.0


#
# speech recognition pipeline
#
# 1. 取特征
# 2. 逐帧分类特征
# 3. 根据概率输出假设
#
# %%
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

SPEECH_FILE = download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H  # pipeline里面很多预训练模型

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)  # 带预训练权重

print(model.__class__)

# VOiCES 数据集

IPython.display.Audio(SPEECH_FILE)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if (sample_rate != bundle.sample_rate):  # 用transform里面的对象而不是functional函数在批量处理时效率更高
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# 上面是在下载模型和数据，同时重采样适应模型输入
with torch.inference_mode():  # 其实 Wav2Vec2 models fine-tuned for ASR task可以一步取特征并分类
    features, _ = model.extract_features( waveform)  # 特征是tensor组成的列表，每个tensor对应一个transformer层输出

fig, ax = plt.subplots(2,1)
# for i, feats in next((features)):
ax[0].imshow(features[0][0].cpu(), interpolation="nearest")
ax[0].set_title(f"Feature from transformer layer {1}")
ax[0].set_xlabel("Feature dimension")
ax[0].set_ylabel("Frame (time-axis)")

with torch.inference_mode():
    emission, _ = model(waveform)  # 输出是logits（没有经过softmax概率化）

plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)
IPython.display.Audio(SPEECH_FILE)

# %%
