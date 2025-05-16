import torch

from PIL import Image

from train import SELFMODEL

from custom_func import data_transforms_func

# 选择计算设备
device = ("cuda" # 优先使用NVIDIA的GPU
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_availbale() # 这是针对apple的加速
          else "cpu")

classes = ('1','2')
model_path = r"/home/taylor/pytorch_introlearning/checkpoints/selfmodels/resnet50_1epoch_acc0.9685_weights.pth"
image_path = r"/home/taylor/pytorch_introlearning/data/test/2/3301.png"

# 对单个目标进行回归
def single_predict(model_path,image_path):
    data_transforms = data_transforms_func(224)['val'] # 设置数据预处理方式
    model = SELFMODEL(len(classes)) # 实例化模型对象
    model = torch.nn.DataParallel(model) # 数据并行化
    weights = torch.load(model_path) # 加载模型参数
    model.load_state_dict(weights) # 参数装载进模型的状态字典
    model.eval() # 设置为评估模式，可禁用梯度跟踪，加快推理
    model = model.to(device) # 模型移到设备（GPU）上
    
    img = Image.open(image_path) # 打开图像文件
    img = data_transforms(img) # 进行数据预处理
    img = img.unsqueeze(0) # 
    img = img.to(device) # 把数据移动到设备上
    output = model(img) # 输入进模型进行推理
    pred_idx = torch.argmax(output).item() # 获取预测概率最大的类的索引
    # _,preds_tensor = torch.max(output,dim=1) # 获取预测概率最大的类的索引
    pred_name = classes[pred_idx]
    print(f"这个是：{pred_name}")
    
if __name__ == '__main__':
    single_predict(model_path,image_path)