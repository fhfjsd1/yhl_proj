# %%
from thop import profile
from torch import nn

import torch
import torchvision
from torchvision.models import resnet18,ResNet18_Weights # 使用pytorch自带的预训练模型和权重
from torchvision.models import mobilenet_v2 # 使用pytorch自带的预训练模型和权重
from torchvision.models import mobilenet_v3_large
from torchinfo import summary as torchinfo_summary

import torch
import torch.nn as nn

class resnet(nn.Module): # 继承所有神经网络的基类
    def __init__(self,out_features=2) -> None:
        super().__init__() # 调用父类的初始化函数
        
        # 使用pytorch自带的预训练模型和权重，这里选用基于 IMAGENET1K_V2 数据集训练的 ResNet50
        self.model_ft = resnet18() 
        num_in_features = self.model_ft.fc.in_features # 获取最后全连接层的输入参数
        self.model_ft.fc = nn.Linear(num_in_features,out_features) # 修改最后一个全连接层输出为本任务类别数
        
        #self.model_ft = mobilenet_v2()

        # self.block = TSConformerBlock(num_channel=3)
        
        # self.vgg11 = torchvision.models.vgg11()
        # self.vgg11.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 2), nn.Softmax(dim=1))

        # num_in_features = self.model_ft.classifier._modules['1'].in_features
        # self.model_ft.classifier._modules['1'] = nn.Linear(num_in_features,out_features)
        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.2)
       
    def forward(self,x): # 前向传播
        # x = torch.nn.functional.relu(self.conv(x))
        logits = self.model_ft(x)
        # x = self.dropout(x) # 在全连接层之前添加 Dropout
        # probabilities = torch.nn.functional.softmax(logits, dim=1)  # 添加 softmax 激活函数
        return logits

class mobile(nn.Module): # 继承所有神经网络的基类
    def __init__(self,out_features=2) -> None:
        super().__init__() # 调用父类的初始化函数
        
        self.model_ft = mobilenet_v2()

        # self.vgg11 = torchvision.models.vgg11()
        # self.vgg11.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 2), nn.Softmax(dim=1))

        num_in_features = self.model_ft.classifier._modules['1'].in_features
        self.model_ft.classifier._modules['1'] = nn.Linear(num_in_features,out_features)
        # 添加 Dropout 层
        # self.dropout = nn.Dropout(p=0.2)
       
    def forward(self,x): # 前向传播
        # x = torch.nn.functional.relu(self.conv(x))
        logits = self.model_ft(x)
        # x = self.dropout(x) # 在全连接层之前添加 Dropout
        # probabilities = torch.nn.functional.softmax(logits, dim=1)  # 添加 softmax 激活函数
        return logits

if __name__ == '__main__':
    model = mobile()

    torchinfo_summary(model, input_size=(1,3,224, 224) , verbose=1)

# %%