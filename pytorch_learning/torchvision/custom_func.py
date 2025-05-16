# 基于蓝图可分离卷积的轻量化花卉识别系统

# Introduction:自定义函数

import math
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms

classes = ('daisy','dandelion','rose','sunflower','tulip') # 分类任务的类别名

# 数据处理函数
def data_transforms_func_pic(img_size=224):
    data_transforms = { # 分为训练集数据处理和验证集数据处理
        'train':
           transforms.Compose([ # 多种方法组合成一个
               transforms.RandomAffine(degrees=0, shear=(0.1, 0.1)), # 随机横向拉伸
               transforms.Resize((img_size,img_size)), # 规范输入尺寸
            #    transforms.RandomRotation((-5,5)), # 随机旋转
               # transforms.RandomAutocontrast(p=0.2), # 随机对比度增强
               transforms.ToTensor(), # 数据转为tensor对象
               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # 像素归一化
           ]),
        'val':
            transforms.Compose([ # 验证集数据不需要进行数据增强
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
    }
    return data_transforms

# 使用余弦退火算法计算新的学习率
def calc_lr(epoch,init_lr,nEpochs,batch=0,nBatches=0):
    t_total = nEpochs * nBatches # 整个训练流程的batch数
    t_cur = epoch * nBatches + batch # 当前batch的global_step
    lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total)) # 余弦退火算法
    return lr

# 更新学习率
def adjust_lr(optimizer,epoch,params,batch=0,nBatches=0):
    new_lr = calc_lr(epoch,params['learning_rate'],
                     params['epochs'],batch,nBatches) # 得到新的学习率
    # pytorch 中有 lr_scheduler.CosineAnnealingLR 用来调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

# 计算正确率
def cal_accuracy(output,labels,num_samples):
    TP_and_TN = 0
    TP_and_TN += ((output.argmax(1))==labels).type(torch.float).sum().item() # 累加每个batch中的正确样本数
    accuracy = TP_and_TN / num_samples
    return accuracy

def add_pr_curve_tensorboard(class_index,probs,label,writer,classes,mode,global_step=0):
        tensorboard_truth = label==class_index # 按类别绘制，选出当前类别索引下的所有该类（即正样本）
        tensorboard_probs = probs[:,class_index] # 所有样本为当前类别索引的概率
        writer.add_pr_curve(classes[class_index]+mode, # 图表名称
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=global_step)
        
def matplotlib_imshow(img,one_channel=False):
    if one_channel: # 是否以灰度图像显示
        img = img.mean(dim=0) # 计算每个像素处通道均值（转为灰度）
    img =img /2 + 0.5 # 去标准化
    img = img.to('cpu')
    npimg =img.numpy()
    if one_channel:
        plt.imshow(npimg,cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg,(1,2,0))) # 彩色图转换通道后再显示

# 绘出每个batch的False样本
def plot_wrongclasses_preds(labels,output,images):
    _,preds_tensor = torch.max(output,dim=1) # 获取预测概率最大的类的索引
    preds_tensor = preds_tensor.to('cpu')
    labels = labels.to('cpu')
    preds = np.squeeze(preds_tensor.numpy()) # 转为numpy数组，再压缩为1维
    labels = np.squeeze(labels.numpy())
    if (preds!=labels).any(): # 如果预测类别的索引与真实标签不同
        probs = [nn.functional.softmax(el,dim=0)[i].item() for i ,el in zip(preds,output)] # 取出预测概率
        fig = plt.figure(figsize=(15,20)) # 创建figure图像对象
        for idx in np.arange(4):
            ax = fig.add_subplot(1,4,idx+1,xticks=[],yticks=[]) # 创建1行4列的子图对象
            matplotlib_imshow(images[idx]) # 显示图像
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format( # 设置子图标题
                classes[preds[idx]], # 预测类别
                probs[idx] * 100.0,
                classes[labels[idx]]), # 真实类别
                color=("green" if preds[idx]==labels[idx].item() else "red"))
        return fig
    return 0
