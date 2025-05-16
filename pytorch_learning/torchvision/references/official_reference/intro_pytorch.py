import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image

device = ("cuda"
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_availbale()
          else "cpu")
print(f"Using {device} device")

class NeuralNetwork(nn.Module): # nn.Module: Base class for all neural network modules.
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device=device)
print(model)
# for name,param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# 加载pytorch自带的一些数据集
training_data = datasets.FashionMNIST(
    root = r"./fasion",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
    # target_transform=transforms.Lambda(lambda y:
    #     torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)

test_data = datasets.FashionMNIST(
    root=r"./fasion",
    train=False,
    download=True,
    transform=transforms.ToTensor()
    # target_transform=transforms.Lambda(lambda y:
    #     torch.zeros(10,dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
)

training_dataloader = DataLoader(training_data,batch_size=64,shuffle=True,)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

learning_rate = 1e-3
batch_size = 64
epochs = 5

def train_loop(dataloader,model,loss_fn,optimiazer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        
        loss.backward()
        optimiazer.step()
        optimiazer.zero_grad()
        
        if batch % 100 == 0:
            loss,current = loss.item(),batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader,model,loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss,correct = 0,0
    
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += ((pred.argmax(1))==y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------")
    train_loop(training_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
print("Done!")

######################################运行上面代码要注释下面的###############################################

train_features,train_labels = next(iter(training_dataloader))
print(train_features)
print(train_labels)
img = train_features[1].squeeze()
label = train_labels[1]
print(img.size())
plt.imshow(img,cmap="gray")
plt.show()
print((label.sum()).item())

# 加载我们自己的一些数据集
class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[index,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label