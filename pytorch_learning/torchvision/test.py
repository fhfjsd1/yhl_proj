from torchvision import datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import bsconv.pytorch # bsconv.pytorch是bsconv的pytorch实现

import os

from tqdm import tqdm

from train import SELFMODEL
from custom_func import *

# 选择计算设备
device = ("cuda" # 优先使用NVIDIA的GPU
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_availbale() # 这是针对apple的加速
          else "cpu")
# 输出一些版本信息，方便环境配置
print(f"Using device: {device} ")
print("pytorch版本：",torch.__version__)
# print(torch.__path__)
print("cuda版本：",torch.version.cuda)
# print(torch.cuda.__path__)
print("cudnn版本：",torch.backends.cudnn.version())

data_path = r"./data/flowers_split" # 数据集路径
writer = SummaryWriter(r'./runs/test') # tensorboard日志输出路径
classes = ('daisy','dandelion','rose','sunflower','tulip') # 分类任务的类别名

# 超参数设置
params = {
    'pretrained_model':'resnet50', # 预训练模型名称
    'img_size':224, # 图片输入大小
    'test_dir':os.path.join(data_path,"test"), # 训练集路径
    'device':device, 
    'batch_size':4, # mini-batch的大小
    'num_workers':2, # 加载数据的子进程数
    'model_path':r"./checkpoints/selfmodels/resnet50_1epoch_acc0.9804_weights.pth", # 存放测试用模型的路径
    'num_classes':len(os.listdir(os.path.join(data_path,"train"))), # 自适应获取类别数目（需以特定文件格式组织数据集）
}

def test_loop(test_loader,model):
    model.eval() # 模型设置为评估模式，加快推理
    num_samples = len(test_loader.dataset)
    stream = tqdm(test_loader)
    acc = 0.0
    class_probs = []
    class_real_label = []
    with torch.no_grad(): # 禁用梯度跟踪，加快速度
        for _,(images,labels) in enumerate(stream,start=1):
            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            output = model(images)
            
            class_probs_batch = [nn.functional.softmax(el,dim=0) for el in output]
            class_probs.append(class_probs_batch)
            class_real_label.append(labels)
    
            acc += cal_accuracy(output,labels,num_samples)
            stream.set_description("Mode: Test.    Accuracy: {acc:.4f}".format(acc=acc))
    
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_real_label = torch.cat(class_real_label)
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i,test_probs,test_real_label,writer,classes,"test")
        
    return acc

if __name__ == '__main__':
    data_transforms = data_transforms_func_pic(params['img_size']) # 设置数据预处理方式
    # 准备数据集，此处使用torchvision自带的加载方式，需按特定文件路径格式组织数据
    test_dataset = datasets.ImageFolder(params['test_dir'],data_transforms['val'])
    test_loader = DataLoader(test_dataset,
                            batch_size=params['batch_size'],
                            shuffle=False,
                            num_workers=params['num_workers'],
                            pin_memory=True) 
    
    model = SELFMODEL(params['num_classes']) # 实例化模型
    replacer = bsconv.pytorch.BSConvU_Replacer() # 替换模型中的卷积层为bsconv
    # 这里的bsconv.pytorch.BSConvU_Replacer()是bsconv的pytorch实现
    model = replacer.apply(model)
    model = nn.DataParallel(model) # 数据并行化，加快速度
    weights = torch.load(params['model_path'])
    model.load_state_dict(weights)
    model = model.to(device) # 模型加载到设备
    
    acc = test_loop(test_loader,model)
    print(acc)
    writer.close()