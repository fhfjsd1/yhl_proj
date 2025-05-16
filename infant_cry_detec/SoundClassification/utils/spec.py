# %%
import io
import random
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio import transforms
from torch.utils.tensorboard import SummaryWriter # 主要用这个进行训练的可视化

import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np
from PIL import Image

def pic_save(melspec,label,id):
    plt.figure()
    librosa.display.specshow(librosa.power_to_db(melspec[0], ref=np.max), sr=8000, fmax=4000)
    # plt.imshow(librosa.power_to_db(melspec[0]), origin="lower", aspect="auto", interpolation="nearest")
    
    output_folder = f'/home/taylor/UrbanSound8k/SoundClassification/data_emo/spec/{label}'  # 替换为你的文件夹路径
    os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
    output_file = os.path.join(output_folder,f'{label}_{id}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()  # 关闭图形窗口，释放内存
    
    # # 将图像保存到内存中
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    
    # # 读取图像并转换为NumPy数组
    # image = np.array(Image.open(buf))
    # buf.close()  # 关闭缓冲区
    # plt.close() # 关闭图形窗口，释放内存
    
    # # 转换为PyTorch张量
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    # return image_tensor, label, id

def plot_spectrogram(specgram,label=None):
    fig = plt.figure(figsize=(20,20)) # 创建figure图像对象
    for idx in np.arange(16):
        ax = fig.add_subplot(4,4,idx+1,xticks=[],yticks=[]) # 创建4行4列的子图对象
        ax.set_title(label[idx])
        # ax.imshow(librosa.power_to_db(specgram[idx][0]), origin="lower", aspect="auto", interpolation="nearest")
    return fig
    
# 数据处理函数
def data_transforms_func(waveform,sample_rate):
    # # 随机截取3到5秒的内容
    # duration = random.uniform(3, 5)  # 随机选择3到5秒
    # num_samples = int(duration * sample_rate)  # 计算对应的样本数
    # max_start = waveform.size(1) - num_samples  # 计算最大起始位置
    # start = random.randint(0, max_start)  # 随机选择起始位置
    # end = start + num_samples  # 计算结束位置
    # waveform = waveform[:, start:end]  # 截取波形

    resampler = transforms.Resample(sample_rate,new_freq=16000,dtype=waveform.dtype,lowpass_filter_width=100)
    resampled_waveform = resampler(waveform)
    # num_frames = resampled_waveform.size(1)
    # if num_frames>144000:
    #     resampled_waveform = resampled_waveform[:,0:144000]
    # else:
    #     pad = 144000 - num_frames
    #     resampled_waveform = torch.nn.functional.pad(resampled_waveform,(0,pad))
    
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        normalized=True,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    
    # melspec = mel_spectrogram(resampled_waveform).numpy()
    melspec = mel_spectrogram(resampled_waveform)
    
    # plt.figure(figsize=(10, 10))
    # librosa.display.specshow(librosa.power_to_db(melspec[0], ref=np.max), sr=8000, fmax=4000)

    # # 将图像保存到内存中
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    
    # # 读取图像并转换为NumPy数组
    # image = np.array(Image.open(buf).convert("RGB"))
    # buf.close()  # 关闭缓冲区
    # plt.close() # 关闭图形窗口，释放内存
    
    # 转换为PyTorch张量
    # image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    
    # tra = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),antialias=True),
    #                                       #torchvision.transforms.ToTensor(),   
    #                                       torchvision.transforms.Normalize((0.5),(0.5)) ])# 像素归一化
    # image_tensor = tra(melspec)
    #  # 规范输入尺寸
    
    # _, ax = plt.subplots(1, 1)
    # ax.imshow(librosa.power_to_db(melspec[0]), origin="lower", aspect="auto", interpolation="nearest")
    # fig = ax.get_figure()
    # fig.canvas.draw()
    # image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # # Step 2: 将 numpy 数组转换为 PIL 图像
    # image_data = torch.tensor(image_data)

    # tra = torchvision.transforms.Resize((224,224))
    # image_data = tra(image_data)
    return melspec

# writer_cry = SummaryWriter(r'./runs/crydata') # tensorboard日志输出路径
annotations_file = r"/home/taylor/UrbanSound8k/SoundClassification/data_emo/metadata/train_with_folds.csv"
audio_dir = r"/home/taylor/UrbanSound8k/SoundClassification/data_emo/audio/train"

# 定义 target_transform 函数
def target_transform(label,classes):
    # 将标签转换为索引
    label_index = classes.index(label)
    # # 创建一个全零的数组
    # one_hot = np.zeros(len(classes), dtype=np.float32)
    # # 将对应索引位置置为1
    # one_hot[label_index] = label_index
    # 转换为 PyTorch 张量
    return torch.tensor(label_index, dtype=torch.long)
    

class CryDateset(Dataset):
    def __init__(self,annotations_file,audio_dir,transform=None,target_transform=None,classes=None) -> None:
        super().__init__()
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        
    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, index):
        audio_path = os.path.join(self.audio_dir,self.audio_labels.iloc[index,1],self.audio_labels.iloc[index,0])
        waveform,sample_rate = torchaudio.load(audio_path)
        label = self.audio_labels.iloc[index,1]
        id = self.audio_labels.iloc[index,2]
        if self.transform:
            waveform = self.transform(waveform,sample_rate)
        if self.target_transform:
            label = self.target_transform(label,self.classes)
        
        return waveform,label,id

train_dataset = CryDateset(annotations_file,audio_dir,transform=data_transforms_func)
train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)

print("训练集总数：",train_dataset.__len__())

for i,([waveform],label,id) in enumerate(iter(train_dataloader)):
    print(waveform.size(),label[0])
    pic_save(waveform,label[0],id.item())
    # fig = plot_spectrogram(waveforms,label=label)
# %%