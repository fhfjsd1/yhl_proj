# %%
import torch
import torch.nn as nn
from torchinfo import summary as torchinfo_summary

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=4, num_classes=2):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256*2, 128, 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        h1 = torch.zeros(4, x.size(0), 128).to(x.device)
        c1 = torch.zeros(4, x.size(0), 128).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm2(out, (h1, c1))
        out = torch.mean(out, 1)
        
        # Decode the hidden state of the last time step
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.softmax(self.fc2(out), dim=1)
        return out

# Hyperparameters
input_size = 128  # 输入特征的维度
hidden_size = 256  # LSTM 隐藏层的维度
num_layers = 4  # LSTM 层数
num_classes = 2  # 类别数（这里是二分类任务）

if __name__ == '__main__':
    # 初始化模型、损失函数和优化器
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

    torchinfo_summary(model, input_size=(1,128, 200) , verbose=1)


# %%
