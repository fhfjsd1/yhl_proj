# %%
from thop import profile
import torch.nn as nn
from speechbrain.utils import hpopt as hp # type: ignore
import speechbrain as sb   # type: ignore

import torch

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN # type: ignore
from speechbrain.lobes.models.ECAPA_TDNN import Classifier # type: ignore
from torchinfo import summary as torchinfo_summary

class ECAPAFullNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = ECAPA_TDNN(
            input_size=128,
            channels=[512, 512, 512, 512, 1024],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=100
        )
        self.classifier = Classifier(
            input_size=100,  # 与 lin_neurons 保持一致
            out_neurons=2
        )

    def forward(self, x):
        # x 形状: (batch, time, n_mels)
        emb = self.embedding_model(x)  # 获得嵌入向量
        out = self.classifier(emb)     # 分类输出
        return out

if __name__ == '__main__':
    # 初始化模型、损失函数和优化器

    model = ECAPAFullNet().eval()

    torchinfo_summary(model, input_size=(1,200, 128) , verbose=1)
# %%