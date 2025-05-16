# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchinfo import summary
import torch.nn.init as init
from thop import profile




# 计算通道均值
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

# 计算通道标准差
def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.0001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
# 点卷积+逐通道
class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

# 通道注意力层
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# 全局响应归一化
class GRN(nn.Module):
    """
    global response normalization as introduced in https://arxiv.org/pdf/2301.00808.pdf
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # dequantize and quantize since torch.norm not implemented for quantized tensors
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

        x = self.gamma * (x * nx) + self.beta + x
        return x

# 蓝图可分卷积层+GRN
class BSConvLayer(nn.Module):
    def __init__(self, input, output, kernel_size, stride=1,padding=1,dilation=1):
        super(BSConvLayer,self).__init__()
        
        self.BSconv = BSConvU(input,output,kernel_size,stride=stride,padding=padding,dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(output)
        self.grn = GRN()
        
    def forward(self,x):
        out = self.BSconv(x)
        if out.size(1)==x.size(1):
            out = self.relu(self.bn(x+out))
        else:
            out = self.relu(self.bn(out))
        return self.grn(out)





class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        #input = input.to(device)
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size,
                 dropout=0, num_layers=1, bidirectional=False):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        # self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                          bidirectional=True))  # intra-segment RNN is always noncausal
            self.col_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        # no output layer in DPCRN
        # self.output = nn.Sequential(nn.PReLU(),
        #                             nn.Conv2d(input_size, output_size, 1)
        #                             )

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        #input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2,
                                                                             1).contiguous()  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1,
                                                                             2).contiguous()  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        # output = self.output(output) # B, output_size, dim1, dim2
        return output
    
class DPCRN(nn.Module):
    def __init__(self, encoder_in_channel, encoder_channel_size, encoder_kernel_size, encoder_stride_size, encoder_padding,
                       decoder_in_channel, decoder_channel_size, decoder_kernel_size, decoder_stride_size,
                       rnn_type, input_size, hidden_size,
                       ):
        super(DPCRN, self).__init__()
        self.encoder_channel_size = encoder_channel_size
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_stride_size = encoder_stride_size
        self.encoder_padding = encoder_padding
        self.decoder_channel_size = decoder_channel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
     

        self.encoder = Encoder(encoder_in_channel, self.encoder_channel_size,
                               self.encoder_kernel_size, self.encoder_stride_size, self.encoder_padding)
        self.decoder = Decoder(decoder_in_channel, self.decoder_channel_size,
                               self.decoder_kernel_size, self.decoder_stride_size)
        self.dprnn = DPRNN(rnn_type, input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):

        inputs = x     # B x C x F x T
        x, skips = self.encoder(inputs)

        x = self.dprnn(x)

        mask = self.decoder(x, skips)
        en_re, en_im = self.mask_speech(mask, inputs)      # en_ shape: B x F x T

        return  en_re, en_im

    def mask_speech(self, mask, x):
        mask_re = mask[:,0,:,:]
        mask_im = mask[:,1,:,:]

        x_re = x[:,0,:,:]
        x_im = x[:,1,:,:]

        en_re = x_re * mask_re - x_im * mask_im
        en_im = x_re * mask_im + x_im * mask_re
        return en_re, en_im    
    
encoder_in_channel = 2
encoder_channel_size = [32,32,64]
encoder_kernel_size = [[5,2],[3,2],[3,2]]
encoder_stride_size = [[2,1],[2,1],[1,1]]
encoder_padding = [[1,0,0,2],[1,0,0,1],[1,0,1,1]]

decoder_in_channel = 128
decoder_channel_size = [32,32,2]
decoder_kernel_size = [[3,2],[3,2],[5,2]]
decoder_stride_size = [[1,1],[2,1],[2,1]]

class Encoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size, padding):
        super(Encoder, self).__init__()
        
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding = padding

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            self.conv.append(nn.Conv2d(in_channels=in_chan,out_channels=channel_size[i],
                                       kernel_size=kernel_size[i], stride=stride_size[i]))
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i]
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x shape: B x C x F x T
        skips = []
        for i, (layer, norm) in enumerate(zip(self.conv, self.norm)):
            x = F.pad(x, pad=self.padding[i])
            x = layer(x)
            x = self.prelu(norm(x))
            skips.append(x)
        return x, skips

class Decoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size):
        super(Decoder, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            if i == 3:
                self.conv.append(nn.ConvTranspose2d(in_channels=in_chan, out_channels=channel_size[i],
                                                    kernel_size=kernel_size[i], stride=stride_size[i],
                                                    padding=[1, 0], output_padding=[1, 0]))
            else:
                self.conv.append(nn.ConvTranspose2d(in_channels=in_chan, out_channels=channel_size[i],
                                                    kernel_size=kernel_size[i], stride=stride_size[i],
                                                    padding=[1,0]))
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i] * 2
        self.prelu = nn.PReLU()

    def forward(self, x, skips):
        # x shape: B x C x F x T
        for i, (layer, norm, skip) in enumerate(zip(self.conv, self.norm, reversed(skips))):
            x =  F.pad(x, pad=(0,0,0,1)) if x.size(2) != skip.size(2) else x
            x = torch.cat([x,skip], dim=1)
            x = layer(x)[:,:,:,:-1]
            x = self.prelu(norm(x))
        return x
    
class ESA(nn.Module):
    def __init__(self, num_feat=32, conv=BSConvU, p=0.25):
        super(ESA, self).__init__()
        
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m
    
    
class Crnn(nn.Module):
    def __init__(self, classes_num=2, activation='logsoftmax',
                 Conv=BSConvLayer):
        super(Crnn, self).__init__()
        
        self.channels = 1
        self.hidden_size = 64
        self.rnn_input_size = 32
        self.num_layers = 1
        self.conv_kernel_size = 32
        self.p = 0.3
        self.activation = activation
        
        self.branch1x1_1 = BasicConv2d(2, 32, kernel_size=1, padding=0)
        # self.readyblock = Conv(1, 32, kernel_size=(2,2), padding=0,stride=2)
         
        
        self.branch3x3_1 = BasicConv2d(2, 32, kernel_size=1)
        self.branch3x3_2 = Conv(32,64, kernel_size=(1, 3), padding=(0, 1)) 
        self.branch3x3_3 = Conv(64,32, kernel_size=(3, 1), padding=(1, 0)) 
        
        self.branch5x5_1 = BasicConv2d(2, 32, kernel_size=1)
        self.branch5x5_2 = Conv(32,64, kernel_size=(1, 5), padding=(0, 2))      
        self.branch5x5_3 = Conv(64,32, kernel_size=(5, 1), padding=(2, 0)) 
        
        self.branch7x7_1 = BasicConv2d(2, 32, kernel_size=1)
        self.branch7x7_2 = Conv(32,64, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = Conv(64,32, kernel_size=(7, 1), padding=(3, 0))      
        
        self.c5 = nn.Conv2d(32*4, 32, kernel_size = 1) 
        self.esa = ESA(32)
        self.cca = CCALayer(32) # 通道注意力层
        self.denoise = DPCRN(encoder_in_channel, encoder_channel_size, encoder_kernel_size, encoder_stride_size, encoder_padding,
                       decoder_in_channel, decoder_channel_size, decoder_kernel_size, decoder_stride_size,
                       rnn_type='LSTM', input_size=encoder_channel_size[-1], hidden_size=encoder_channel_size[-1])

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=(256, 1), stride=1) #97
        
        # 全局最大池化层
        self.global_max_pool = nn.AdaptiveMaxPool2d((96, 1))

        # rnn-gru layer
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=self.p
        )
        self.extractor = SELFMODEL()
        # fully connected layer
        self.fc = nn.Sequential(
            # nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Linear(32, self.hidden_size),
            nn.Dropout(self.p),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, classes_num),
            nn.Sigmoid()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def init_hidden(self, batch_size, hidden_size):
        return Variable(torch.zeros(self.num_layers*2, batch_size, hidden_size)).cuda()

    def forward(self, x):

        x,y = self.denoise(x)

        
        x = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),1)
        x = x[:,:,0:256,:]
        output = self.extractor(x)
        # [batch, channel, 256, 313]
        # x = x.unsqueeze(1)

        # branch1x1 = self.branch1x1_1(x)         

        # branch3x3 = self.branch3x3_1(x)  
        # branch3x3 = self.branch3x3_2(branch3x3)
        # branch3x3 = self.branch3x3_3(branch3x3)
        
        # branch5x5 = self.branch5x5_1(x) 
        # branch5x5 = self.branch5x5_2(branch5x5)
        # branch5x5 = self.branch5x5_3(branch5x5)
               
        
        # branch7x7 = self.branch7x7_1(x)
        # branch7x7 = self.branch7x7_2(branch7x7)
        # branch7x7 = self.branch7x7_3(branch7x7)

        # outputs = (branch1x1,branch3x3,branch5x5,branch7x7)
        # feature = torch.cat(outputs, 1)
        # feature = self.c5(feature)
        # # 添加数值检查
        # if torch.isnan(feature).any() or torch.isinf(feature).any():
        #     print(feature)
        #     raise ValueError("NaN or Inf detected in feature before CCA")

        # feature = self.esa(feature)
        # feature = self.cca(feature) + branch1x1

        # # 添加数值检查
        # if torch.isnan(feature).any() or torch.isinf(feature).any():
        #     raise ValueError("NaN or Inf detected in feature after CCA")

        # output = self.max_pool(feature).squeeze(-2)
        # output = output.permute(0, 2, 1)
        # # output = self.global_max_pool(output).squeeze(-1)
        # # 展平特征图
        # # feature = feature.contiguous().view(batch_size, -1) 
    
        # # h_state = self.init_hidden(batch_size, self.hidden_size)
        # # self.rnn.flatten_parameters()
        # # output, h_state = self.rnn(output, h_state)
        
        # # output = output[:, :, self.hidden_size:]
        # # output = output[:, -1, :].squeeze(1)
        # output = torch.mean(output, dim=1)
        
        # output = self.fc(output)
        
        return output  


from torchvision.models import mobilenet_v2 # 使用pytorch自带的预训练模型和权重

class SELFMODEL(nn.Module): # 继承所有神经网络的基类
    def __init__(self,out_features=2) -> None:
        super().__init__() # 调用父类的初始化函数
        
        # 使用pytorch自带的预训练模型和权重，这里选用基于 IMAGENET1K_V2 数据集训练的 ResNet50
        # self.model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        # num_in_features = self.model_ft.fc.in_features # 获取最后全连接层的输入参数
        # self.model_ft.fc = nn.Linear(num_in_features,out_features) # 修改最后一个全连接层输出为本任务类别数
        
        self.model_ft = mobilenet_v2()
        self.conv = nn.Conv2d(2,3,1)
        # self.block = TSConformerBlock(num_channel=3)
        
        # self.vgg11 = torchvision.models.vgg11()
        # self.vgg11.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 2), nn.Softmax(dim=1))

        num_in_features = self.model_ft.classifier._modules['1'].in_features
        self.model_ft.classifier._modules['1'] = nn.Linear(num_in_features,out_features)
        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.2)
       
    def forward(self,x): # 前向传播
        x = torch.nn.functional.relu(self.conv(x))
        logits = self.model_ft(x)
        # x = self.dropout(x) # 在全连接层之前添加 Dropout
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # 添加 softmax 激活函数
        return logits
    
if __name__ == '__main__':
    model = Crnn()
   
    # summary(model, input_size=(1,256, 313), verbose=0)
    
    en = Encoder(encoder_in_channel, encoder_channel_size, encoder_kernel_size, encoder_stride_size, encoder_padding)
    # outen = en(torch.randn(4, 2, 257, 79))
    de = Decoder(decoder_in_channel, decoder_channel_size, decoder_kernel_size, decoder_stride_size)

    dpcrn = DPCRN(encoder_in_channel, encoder_channel_size, encoder_kernel_size, encoder_stride_size, encoder_padding,
                       decoder_in_channel, decoder_channel_size, decoder_kernel_size, decoder_stride_size,
                       rnn_type='LSTM', input_size=64, hidden_size=64)
    
   # out = dpcrn(torch.randn(4, 2, 201, 200))
    
    input_size = torch.randn(1, 2, 257, 101)  # 输入张量的形状 (batch_size, channels, height, width)
    flops, params = profile(model, inputs=(input_size,))
    print(f"FLOPs: {flops/10**9:.2f}G, Params: {params/10**6:.2f}M")

    
    flops, params = profile(dpcrn, inputs=(torch.randn(1, 2, 257, 101),))
    print(f"FLOPs: {flops/10**9:.2f}G, Params: {params/10**6:.2f}M")

# %%
