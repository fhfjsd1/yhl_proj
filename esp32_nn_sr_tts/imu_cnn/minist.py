import torch
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from PIL import Image

import torch.nn as nn
import torch.optim as optim

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations: MNIST is grayscale so convert to 3 channels, resize to 224x224 and normalize with Imagenet means/std
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # Load training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = models.mobilenet_v3_small(weights= MobileNet_V3_Small_Weights.DEFAULT )
    num_ftrs = model.classifier[3].in_features  # the Linear layer is at index 1
    model.classifier[3] = nn.Linear(num_ftrs, 10)

    model = model.to(device)
    # model.load_state_dict(torch.load("mobilenet_mnist_best.pth"))
    # 加载保存的图像文件 (确保文件路径正确)
    # img_path = "trajectory.png"
    # try:
    #     image = Image.open(img_path).convert("RGB")
    # except Exception as e:
    #     print("无法加载图像:", e)
    #     exit(1)

    # input_tensor = transform(image).unsqueeze(0)


    # # 设置模型为评估模式并执行推理
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(input_tensor)
    #     _, predicted = torch.max(outputs, 1)
        
    # print("预测的类别为:", predicted.item())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    model.train()
    best_accuracy = 0.0
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # Evaluate on test set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch+1}/{num_epochs}] Accuracy on test set: {100 * correct / total:.2f}%")
        if correct / total > best_accuracy:
            best_accuracy = correct / total
            # Save the model if it has improved
            torch.save(model.state_dict(), "mobilenet_mnist_best.pth")
        model.train()

if __name__ == "__main__":
    main()