import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.version
import torchvision
from torchvision import transforms

from torch import nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device(
     "cuda"  if torch.cuda.is_available()
     else    "cpu"
)
print(f"Use Device: {device}")
print(torch.__version__)
print(torch.__path__)
print(torch.version.cuda)
print(torch.cuda.__path__)
print(torch.backends.cudnn.version())

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,),(0.5,))]
)

trainset = torchvision.datasets.FashionMNIST(
    root='./fasion',
    download=True,
    train=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    root='./fasion',
    download=True,
    train=False,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def matplotlib_imshow(img,one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img =img /2 + 0.5
    img = img.to('cpu')
    npimg =img.numpy()
    if one_channel:
        plt.imshow(npimg,cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg,(1,2,0)))

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

dataiter = iter(trainloader)
images,labels = next(dataiter)
images,labels = images.to(device),labels.to(device)
img_grid = torchvision.utils.make_grid(images)    

writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(net,images)

def select_n_random(data,labels ,n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n],labels[perm][:n]

def images_to_probs(net,images):
    output = net(images)
    _,preds_tensor = torch.max(input=output,dim=1)
    preds_tensor = preds_tensor.to('cpu')
    preds = np.squeeze(preds_tensor.numpy())
    return preds,[F.softmax(el,dim=0)[i].item() for i ,el in zip(preds,output)]

def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
 
running_loss = 0.0
for epoch in range(1):
    for i,data in enumerate(trainloader,0):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        ouputs = net(inputs)
        loss = criterion(ouputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 1000 == 999:
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net,inputs,labels),
                              global_step=epoch*len(trainloader)+i)
            running_loss =0.0

class_probs = []
class_label =[]
with torch.no_grad():
    for data in testloader:
        images,labels = data
        images,labels = images.to(device),labels.to(device)
        output = net(images)
        class_probs_batch = [F.softmax(el,dim=0) for el in output]
        class_probs.append(class_probs_batch)
        class_label.append(labels)
        
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

def add_pr_curve_tensorboard(class_index,test_probs,test_label,global_step=0):
    tensorboard_truth = test_label==class_index
    tensorboard_probs = test_probs[:,class_index]
    
    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

for i in range(len(classes)):
    add_pr_curve_tensorboard(i,test_probs,test_label)

torch.save(net.state_dict(),r'./weights.pth')
print("finished training")


            
