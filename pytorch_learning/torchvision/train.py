# 基于蓝图可分离卷积的轻量化花卉识别系统
# 核心参考：Pytorch官方文档（https://pytorch.org/tutorials）
# 数据集：https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data

# Introduction：神经网络的训练

# 引入相关库及依赖项
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.utils
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter # 主要用这个进行训练的可视化
from torch import nn

from torchvision import datasets
from torchvision.models import resnet34,ResNet34_Weights # 使用pytorch自带的预训练模型和权重

import os

from tqdm import tqdm # 进度条可视化

from custom_func import * # 自己写的一些函数

import sys
import bsconv.pytorch # bsconv.pytorch是bsconv的pytorch实现

print(sys.path)
import random
import numpy as np
import torch
from torchinfo import summary
from thop import profile

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

# 选择计算设备
device = ("cuda" # 优先使用NVIDIA的GPU
          if torch.cuda.is_available()
          # else "mps" if torch.backends.mps.is_availbale() # 这是针对apple的加速
          else "cpu")

# # 输出一些版本信息，方便环境配置
# print(f"Using device: {device} ")
# print("pytorch版本：",torch.__version__)
# # print(torch.__path__)
# print("cuda版本：",torch.version.cuda)
# # print(torch.cuda.__path__)
# print("cudnn版本：",torch.backends.cudnn.version())

data_path = r"./data/flowers_split" # 数据集路径
writer = SummaryWriter(r'./runs') # tensorboard日志输出路径
# writer_val = SummaryWriter(r'./runs/val') # 分两个writer分别写训练和验证日志
classes = ('daisy','dandelion','rose','sunflower','tulip') # 分类任务的类别名

# 超参数字典
params = {
    'pretrained_model':'resnet34', # 预训练模型名称
    'img_size':224, # 图片输入大小（resnet50输入层为224*224*3）
    'train_dir':os.path.join(data_path,"train"), # 训练集路径
    'val_dir':os.path.join(data_path,"validation"), # 验证集路径
    'device':device, 
    'learning_rate':1e-4, # 初始学习率
    'batch_size':16, # mini-batch的大小
    'num_workers':4, # 加载数据的子进程数
    'epochs':500, # 遍历数据集的次数
    'save_dir':r"./checkpoints/", # 存放模型相关的路径
    'num_classes':len(os.listdir(os.path.join(data_path,"train"))), # 自适应获取类别数目（需以特定文件格式组织数据集）
    'weight_decay':1e-5 # 学习衰减率
}

# 环境变量设置
os.environ['TORCH_HOME'] = params['save_dir'] # 预训练模型下载路径

# 模型定义
class SELFMODEL(nn.Module): # 继承所有神经网络的基类
    def __init__(self,out_features=params['num_classes']) -> None:
        super().__init__() # 调用父类的初始化函数
        
        # 使用pytorch自带的预训练模型和权重，这里选用基于 IMAGENET1K_V2 数据集训练的 ResNet50
        self.model_ft = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) 
        num_in_features = self.model_ft.fc.in_features # 获取最后全连接层的输入参数
        self.model_ft.fc = nn.Linear(num_in_features,out_features) # 修改最后一个全连接层输出为本任务类别数
        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.4)
       
    def forward(self,x): # 前向传播
        logits = self.model_ft(x)
        x = self.dropout(x) # 在全连接层之前添加 Dropout
        return logits

# 定义训练流程
def train_loop(train_loader,model,criterion,optimizer,epoch):
    model.train() # 模型设置为训练模式
    nBatches = len(train_loader) # 数据集的总batch数
    num_samples = len(train_loader.dataset) # 总样本数
    stream = tqdm(train_loader) # 转为tqdm对象，用于绘制进度条
    acc = 0.0 # 本次循环（一个epoch）结束的正确率
    running_loss = 0.0 # 用于计算每个监测段的平均损失
    epoch_loss = 0.0 # 本次循环（一个epoch）结束的平均损失
    class_probs = [] # 预测结果的置信度
    class_label = [] # 真实的标签
    
    # 开始迭代训练
    for batch,(images,labels) in enumerate(stream, start=1):
        images = images.to(device,non_blocking=True) # 异步传输数据放到GPU
        labels = labels.to(device,non_blocking=True)
        output = model(images) # 前馈传播
        loss = criterion(output,labels) # 计算损失函数
        optimizer.zero_grad() # 梯度清零（因为梯度默认累加）
        loss.backward() # backward propagation
        optimizer.step() # 优化器更新权重参数
        adjust_lr(optimizer,epoch,params,batch,nBatches) # 调整学习率
        
        # 下面主要是训练时的数据记录
        class_probs_batch = [nn.functional.softmax(el,dim=0) for el in output] # 这个batch的预测概率
        class_probs.append(class_probs_batch) # 加到列表
        class_label.append(labels) # 把这次batch的真实标签也加到列表
        running_loss += loss.item() # 累加每次都损失函数值
        epoch_loss += loss.item() # 也是累加损失
        acc += cal_accuracy(output,labels,num_samples) # 计算总的正确率
        # fig = plot_wrongclasses_preds(labels,output,images)
        # if fig != 0: # 绘出每个batch的False样本
        #     writer_train.add_figure('wrong',fig,global_step=(epoch-1)*nBatches+batch)
        stream.set_description( # 更新进度条
        "Epoch: {epoch}.    Mode: Train.    Accuracy: {acc:.4f}| Loss: {loss:.6f}  ".format(
            epoch=epoch,acc=acc,loss=loss))
    
        if batch % nBatches == 0: # 每一百个batch记录一次平均损失
            writer.add_scalar("loss/training",running_loss/nBatches,(epoch-1)*nBatches+batch)
            writer.add_scalar("acc/training",acc,(epoch-1)*nBatches+batch)
            running_loss = 0.0
    
    # 绘制precision-recall曲线评估模型性能
    train_probs = torch.cat([torch.stack(batch) for batch in class_probs]) # 拼接预测概率为tensor
    train_label = torch.cat(class_label) # 拼接标签为tensor
    for i in range(len(classes)): # 对所有类别迭代，绘制每个类别的图像
        add_pr_curve_tensorboard(i,train_probs,train_label,writer,classes,"train",epoch)
        
    return acc,epoch_loss/nBatches # 返回这个epoch的accuracy和loss

# 定义验证流程，和训练流程很像
def validation_loop(val_loader,model,criterion,epoch):
    model.eval() # 模型设置为评估模式，加快推理
    nBatches = len(val_loader)
    num_samples = len(val_loader.dataset)
    stream = tqdm(val_loader)
    acc = 0.0
    running_loss = 0.0
    epoch_loss = 0.0
    class_probs = []
    class_label = []
    with torch.no_grad(): # 禁用梯度跟踪，加快速度
        for batch,(images,labels) in enumerate(stream,start=1):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            output = model(images)
            loss = criterion(output,labels)
            
            class_probs_batch = [nn.functional.softmax(el,dim=0) for el in output]
            class_probs.append(class_probs_batch)
            class_label.append(labels)
            running_loss += loss.item()
            epoch_loss += loss.item()
            acc += cal_accuracy(output,labels,num_samples)
            stream.set_description(
            "Epoch: {epoch}.    Mode: Validation.    Accuracy: {acc:.4f}| Loss: {loss:.6f}  ".format(
                epoch=epoch,acc=acc,loss=loss))
    
            if batch % nBatches == 0:
                writer.add_scalar("loss/validation",running_loss/nBatches,(epoch-1)*nBatches+batch)
                writer.add_scalar("acc/validation",acc,(epoch-1)*nBatches+batch)
                # with SummaryWriter("./runs/train2") as writer:
                #     writer.add_scalar("validation",running_loss/10,(epoch-1)*nBatches+batch)
                running_loss = 0
    
    val_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    val_label = torch.cat(class_label)
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i,val_probs,val_label,writer,classes,"val",epoch)
        
    return acc,epoch_loss/nBatches

if __name__ == '__main__':
    # 输出一些版本信息，方便环境配置
    print(f"Using device: {device} ")
    print("pytorch版本：",torch.__version__)
    # print(torch.__path__)
    print("cuda版本：",torch.version.cuda)
    # print(torch.cuda.__path__)
    print("cudnn版本：",torch.backends.cudnn.version())
    
    data_transforms = data_transforms_func_pic(params['img_size']) # 设置数据预处理方式
    # 准备数据集，此处使用torchvision自带的加载方式，需按特定文件路径格式组织数据
    train_dataset = datasets.ImageFolder(params['train_dir'],data_transforms['train']) 
    val_dataset = datasets.ImageFolder(params['val_dir'],data_transforms['val']) 
    
    # 加载内置数据集FashionMNIST
    # train_dataset = torchvision.datasets.FashionMNIST(
    #                     root='./fasion',
    #                     download=True,
    #                     train=True,
    #                     transform=data_transforms)
    # val_dataset = torchvision.datasets.FashionMNIST(
    #                     root='./fasion',
    #                     download=True,
    #                     train=False,
    #                     transform=data_transforms)
    
    
    train_loader = DataLoader(train_dataset,
                            batch_size=params['batch_size'],
                            shuffle=True, # 每次加载时自动打乱
                            num_workers=params['num_workers'], # 加载数据的子进程数
                            # pin_memory=True, # page-locked固定数据内存，加快速度
                            drop_last=False) 
    val_loader = DataLoader(val_dataset,
                            batch_size=params['batch_size'],
                            shuffle=False,
                            num_workers=params['num_workers'],
                            # pin_memory=True
                            drop_last=False)
    
    

    # annotations_file_train = r"/home/taylor/pytorch_introlearning/train.csv"
    # annotations_file_val = r"/home/taylor/pytorch_introlearning/val.csv"
    # audio_dir_train = r"/home/taylor/cry_data/data_emo/train"
    # audio_dir_val = r"/home/taylor/cry_data/data_emo/val"
    # train_dataset = CryDateset(annotations_file_train, audio_dir_train, 
    #                            transform=data_transforms_func,target_transform=target_transform,classes=classes)
    # train_loader = DataLoader(train_dataset,batch_size=params['batch_size'],shuffle=True,num_workers=2)
    # val_dataset = CryDateset(annotations_file_val, audio_dir_val, 
    #                          transform=data_transforms_func,target_transform=target_transform,classes=classes)
    # val_loader = DataLoader(val_dataset,batch_size=params['batch_size'],shuffle=False,num_workers=2)
    
    # print("data classes:",train_dataset.classes)
    print("training data:",len(train_dataset))
    print("validation data:",len(val_dataset))
    print("data is ready!\n")
    
    model = SELFMODEL(params['num_classes']) # 实例化模型
    # 计算 FLOPs 和总参数量
    dummy_input = torch.randn(1, 3, params['img_size'], params['img_size'])
    flops, total_params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"标准：Total Params: {total_params/1e6:.2f} M")
    print(f"标准：Total FLOPs: {flops/1e9:.2f} G")
    
    replacer = bsconv.pytorch.BSConvU_Replacer() # 替换模型中的卷积层为bsconv
    # 这里的bsconv.pytorch.BSConvU_Replacer()是bsconv的pytorch实现
    model = replacer.apply(model)

    # 计算 FLOPs 和总参数量
    dummy_input = torch.randn(1, 3, params['img_size'], params['img_size'])
    flops, total_params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"蓝图：Total Params: {total_params/1e6:.2f} M")
    print(f"蓝图：Total FLOPs: {flops/1e9:.2f} G")
    
    model = nn.DataParallel(model) # 数据并行化，加快速度
    model = model.to(device) # 模型加载到设备
    criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数评估模型
    optimizer = torch.optim.AdamW(model.parameters(),params['learning_rate'],
                                  weight_decay=params['weight_decay']) # 使用AdamW权重优化算法
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    # 绘制模型结构图
    dataiter = iter(train_loader)
    images,labels = next(dataiter)
    images,labels = images.to(device),labels.to(device)
    writer.add_graph(model,images)
    
    save_dir = os.path.join(params['save_dir'],"selfmodels") # 模型的保存路径
    if not os.path.isdir(save_dir): # 如果路径不存在就创建
        os.makedirs(save_dir)
        print("Save diretory {0:} is created".format(save_dir))
    
    best_acc = 0.0 # 用于计算最好的正确率
    previous_save_path = None # 用于存储上一次保存模型的路径
    for epoch in range(1,params['epochs']+1): # 迭代epochs次
        acc,loss = train_loop(train_loader,model,criterion,optimizer,epoch) 
        val_acc,val_loss = validation_loop(val_loader,model,criterion,epoch)
        # scheduler.step(val_loss) # 使用验证损失更新学习率调度器
        print("new learning rate:",optimizer.param_groups[0]['lr'])
        if val_acc >= best_acc: # 如果该次在验证集上的正确率更高，就保存模型
            save_path = os.path.join(save_dir,
                                     f"{params['pretrained_model']}_{epoch}epoch_acc{val_acc:.4f}_weights.pth")
            if previous_save_path and os.path.exists(previous_save_path):
                os.remove(previous_save_path) # 删除上一次保存的模型
        
            torch.save(model.state_dict(),save_path) # 仅保存模型的权重
            best_acc = val_acc
            previous_save_path = save_path # 更新上一次保存模型的路径
            
    writer.close()
    # writer_val.close()
    print("Done!!!!!")         