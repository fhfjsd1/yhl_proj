# %%

# 一些搭建模型的组件

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile

from functools import partial
from timm.models.layers import DropPath, trunc_normal_

from einops.layers.torch import Rearrange

# 通道混洗
def channle_shuffle(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x

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

# 堆叠相同块
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# 逐通道+点卷积
class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(  # 逐通道卷积
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(  # 点卷积，减少通道数
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out

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

# 增强空间注意力
class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
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

# 使用了通道注意力层，特征蒸馏的模型
class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ESDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1) # 蒸馏通道
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, conv)
        self.cca = CCALayer(in_channels) # 通道注意力层
        
    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out)
        return out_fused + input # 残差连接

# ESDB的堆叠
class BSRNLayer(nn.Module):
    def __init__(self,  num_feat=16, num_block=4, num_out_ch=3, upscale=4,
                 conv=BSConvU, p=0.25):
        super(BSRNLayer, self).__init__()
        
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        print(conv)

        self.conv = conv
        #self.fea_conv = self.conv(num_feat*4, num_feat, kernel_size=3, **kwargs)

        self.B1 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        #self.B5 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
       

        self.c1 = nn.Conv2d(num_feat*num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
    

    def forward(self, input):
        #input = torch.cat([input, input, input, input], dim=1)
        #out_fea = self.fea_conv(input)
        out_B1 = self.B1(input)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        #out_B5 = self.B5(out_B4)


        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        output = self.c2(out_B) + input
        #output = self.c2(out_B) + out_fea

        return output

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
    def __init__(self, input, output):
        super(BSConvLayer,self).__init__()
        self.BSconv = BSConvU(input,output,3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(input)
        self.grn = GRN()
        
    def forward(self,x):
        out = self.BSconv(x)
        out = self.relu(self.bn(x+out))
        return self.grn(out)

# 基础卷积块：包括卷积、BN和ReLU
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.0001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
# MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 多头注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 注意力+MLP
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

       
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H=32, W=8):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return x_r#F.interpolate(x_r, size=(H * 2, W * 2))

class FCU(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCU, self).__init__()

       
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H=16, W=4):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * 2, W * 2))

# 几个卷积块(ResNet可选)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x
    
class Model(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        #self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(2, 32, 5, stride=1, padding=2)
        self.conv2 = BSRNLayer(num_feat=32)
        
        self.conv7 = nn.Conv2d(32,32,3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(32, eps=1e-03, momentum=0.01)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.maxpool1 = nn.MaxPool2d((3,2))
        self.maxpool2 = nn.MaxPool2d((6,4))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(3584, n_hidden) 
        self.l2 = nn.Linear(n_hidden, n_output)
        #self.shortcut = nn.quantized.FloatFunctional()
        #self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        #x = self.quant(x)
        out = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(out)
        out = self.maxpool1(x)
        #x = self.maxpool1(x) 
        
        #out = self.maxpool1(x)
        x = self.conv7(out)
        x = self.dropout(self.maxpool2(x))
        x = self.flatten(x)
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x

class Conformer(nn.Module):

    def __init__(self, patch_size=10, in_chans=2, num_classes=2, base_channel=16, channel_ratio=2, num_med_block=0,
                 dim=28, depth=3, num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.dim =dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        #self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.pooling = nn.AdaptiveAvgPool2d((3,1))
        self.conv_cls_head = nn.Linear(dim*6, num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv2 = nn.Conv2d(dim,dim,kernel_size=1,stride=1,padding=0)
        self.bn1 =nn.BatchNorm2d(dim)
        self.bn2=nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(2,dim,kernel_size=2,stride=2,padding=0)

        
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))  
        #self.dropout = nn.Dropout(p=0.03)

        self.conv3 = BSConvLayer(dim,dim)
        self.conv4 = BSConvLayer(dim,dim)
        self.trans_patch_conv = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=0,groups=4)
        
        self.trans_1 = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate)

        self.up = FCU(dim,dim)
        self.shuffle = channle_shuffle
        
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # pdb.set_trace()
        # stem stage [N, 2, 256, 65] -> [N, 15, 64, 16]
        x_base1=self.act1(self.bn1(self.conv1(x)))
        x_base1 = self.maxpool(x_base1)
        x_base2 =self.act1(self.bn2(self.conv2(x_base1)))
        x_base = self.maxpool(self.conv3(x_base1+x_base2)) #[N,16,21,11]
        # 1 stage
        x = self.conv4(x_base)
        x_t = self.trans_patch_conv(x_base)
        x_t = self.shuffle(x_t,2)
        x_t=x_t.flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        
        #特征交流
        X_T = self.up(x_t)
        NEW = torch.cat([X_T,x],dim=1)
        NEW = self.shuffle(NEW,4)

        x_p = self.pooling(NEW).flatten(1)
        conv_cls = self.conv_cls_head(x_p)
        return conv_cls

class Model1(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1,32))
        # Classifier head
        self.pooling = nn.AdaptiveAvgPool2d((3,1))
        self.conv_cls_head = nn.Linear(32*6, 10)
        
        self.conv1 = nn.Conv2d(2, 32, 2, stride=2, padding=0)
        self.conv2 = BSConvLayer(32,32)
        self.conv3 = BSConvLayer(32,32)
        self.conv7 = nn.Conv2d(32,32,1,stride=1,padding=0)
        self.bn =nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
       
        self.maxpool = nn.MaxPool2d((2,2))

        self.trans_patch_conv = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.trans_1 = Block(dim=32, num_heads=4, mlp_ratio=2., qkv_bias=False,
                             qk_scale=None, drop=0., attn_drop=0.)

        self.up = FCUUp(32,32)
        self.shuffle = channle_shuffle
        #self.flatten = nn.Flatten()
        #self.l1 = nn.Linear(3584, n_hidden) 
        #self.l2 = nn.Linear(n_hidden, n_output)
        
        
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.relu(self.bn(self.conv1(x)))
        out =self.maxpool(x)
        
        x = self.relu(self.bn2(self.conv7(out)))
        x1 =self.conv2(x+out)
        x2 = self.conv3(self.maxpool(x1))
        x_t = self.trans_patch_conv(x1)
        x_t=x_t.flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        X_T = self.up(x_t)
        NEW = torch.cat([X_T,x2],dim=1)
        NEW = self.shuffle(NEW,6)
        x_p = self.pooling(NEW).flatten(1)
        conv_cls = self.conv_cls_head(x_p)
        #x = self.flatten(x)
        #x = self.relu(self.l1(x))
        #x = self.dropout(x)
        #x = self.l2(x)
        return conv_cls

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)

class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)

class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=3, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=3, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

#Two-stage conformer (TS-Conformer)
class TSConformerBlock(nn.Module):
    def __init__(self, num_channel=64):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, n_head=3, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel, n_head=3, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x

# %%
