import os
import re
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, TensorDataset, random_split # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore

# 动作分类名
motion_names = ['RightAngle', 'SharpAngle', 'Lightning', 'Triangle', 'Letter_h', 
                'letter_R', 'letter_W', 'letter_phi', 'Circle', 'UpAndDown', 'Horn', 'Wave', 'NoMotion']

# 定义目录路径
DEF_SAVE_TO_PATH = r'./TraningData_3_19/'
DEF_MODEL_NAME = 'model.h5'
DEF_MODEL_H_NAME = 'weights.h'
DEF_FILE_MAX = 130
DEF_MAX_TRIALS = 3
DEF_N_ROWS = 150
#DEF_COLUMNS = (0, 1, 2, 3, 4, 5)
DEF_COLUMNS = (3, 4, 5)
#DEF_COLUMNS = (0, 1, 2)

# 文件格式
DEF_FILE_FORMAT = '.txt'
# 文件名分隔符
DEF_FILE_NAME_SEPERATOR = '_'
DEF_BATCH_SIZE = 60
DEF_NUM_EPOCH = 1600

# 动作名称到标签的映射
motion_to_label = {name: idx for idx, name in enumerate(motion_names)}

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 30, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(30, 20, kernel_size=3, stride=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=15, stride=1)
        self.fc1 = nn.Linear(50*20, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        # x = self.maxpool1(x).squeeze(-1)
        x = torch.sigmoid(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train(model, train_loader, val_loader, num_epochs=DEF_NUM_EPOCH, learning_rate=0.001, early_stopping_patience=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=100, verbose=True)
    
    best_val_accuracy = 0
    best_model = None
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                ground_truth = torch.argmax(y_batch, 1)
                total += y_batch.size(0)
                correct += (predicted == ground_truth).sum().item()

        val_accuracy = correct / total
        scheduler.step(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    return best_model, best_val_accuracy

# 加载数据集
def load_dataset(root_dir, max_rows=None):
    file_list = []
    labels = []
    for filename in os.listdir(root_dir):
        if filename.endswith(DEF_FILE_FORMAT):
            match = re.match(rf'^([\w]+)_([\d]+){DEF_FILE_FORMAT}$', filename)
            if match:
                motion_name = match.group(1)
                number_str = match.group(2)
                number = int(number_str)
                if 0 <= number <= DEF_FILE_MAX:
                    if motion_name in motion_to_label:
                        file_path = os.path.join(root_dir, filename)
                        # 使用max_rows参数限制读取的行数
                        data = np.loadtxt(file_path, delimiter=' ', usecols=DEF_COLUMNS, max_rows=max_rows)
                        file_list.append(data)
                        labels.append(motion_to_label[motion_name])
                    else:
                        print(f"Motion name not recognized: {filename}")
                else:
                    print(f"Number out of range: {filename}")
            else:
                print(f"Invalid file name format: {filename}")
    return file_list, labels


if __name__ == '__main__':
    file_list, labels = load_dataset(DEF_SAVE_TO_PATH, max_rows=DEF_N_ROWS)

    # 数据预处理，例如填充序列以达到统一长度
    max_len = max([len(x) for x in file_list])  # 找到最长序列的长度
    print(f"Max length of sequences: {max_len}")  # 打印max_len的值
    # file_list_padded = pad_sequences(file_list, maxlen=max_len, dtype='float32', padding='post', value=0)

    # 转换标签为one-hot编码
    num_classes = len(motion_names)
    labels_one_hot = np.eye(num_classes)[labels]

    # 转换为PyTorch张量
    x_tensor = torch.tensor(file_list, dtype=torch.float32)
    y_tensor = torch.tensor(labels_one_hot, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(x_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=DEF_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DEF_BATCH_SIZE, shuffle=False)

    # 初始化模型
    input_shape = (DEF_N_ROWS, 3)
    model = CNNModel(input_shape, len(motion_names))

    # 训练模型并保存最佳模型
    best_model, best_val_accuracy = train(model, train_loader, val_loader)

    # 使用最佳模型进行最终评估
    model.load_state_dict(best_model)
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.argmax(y_batch, dim=1)
            y_true_list.extend(true_labels.numpy())
            y_pred_list.extend(predicted.numpy())

    import matplotlib.pyplot as plt # type: ignore
    cm = confusion_matrix(y_true_list, y_pred_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=motion_names)
    disp.plot(cmap=plt.cm.Oranges)
    plt.title(f"Confusion Matrix (Best Val Accuracy: {best_val_accuracy:.2%})")
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    input_sample = torch.rand(1,150,3)
    torch.onnx.export(model,input_sample,'model.onnx',input_names=["input_names"],output_names=["output_names"] )

    # # 确保 target_names 的长度与实际类别数相匹配
    # unique_classes = np.unique(np.concatenate((torch.argmax(y_true, 1), y_pred)))
    # target_names = [motion_names[i] for i in sorted(unique_classes)]

    # 打印每个类别的准确率
    #print(classification_report(y_true, y_pred, target_names=target_names))