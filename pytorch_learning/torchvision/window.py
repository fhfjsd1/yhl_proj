# 基于蓝图可分离卷积的轻量化花卉识别系统
# 作者：电子与信息学院 22级信息工程1班 于浩林 王承志

# Introduction:GUI

import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys
import cv2
import torch
import os
from train import SELFMODEL

from custom_func import data_transforms_func
from PIL import Image

# 选择计算设备
device = ("cuda" # 优先使用NVIDIA的GPU
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_availbale() # 这是针对apple的加速
          else "cpu")

class MainWindow(QTabWidget):
    def __init__(self) -> None:
        super().__init__() # 调用父类初始化函数
        self.setWindowTitle('基于Pytorch的花卉识别系统') # 设置窗口标题
        self.resize(1200, 800) # 设置窗口大小
        self.setWindowIcon(QIcon(r"/home/taylor/pytorch_introlearning/data/flowers/sunflower/40410814_fba3837226_n.jpg")) # 设置图标
        # 图片读取进程
        self.output_size = 480 # 图片输出展示的大小
        self.img2predict = "" # 预测结果
        self.origin_shape = ()
        
        # 模型准备
        self.model_path = r"/home/taylor/pytorch_introlearning/checkpoints/selfmodels/resnet50_10epoch_acc0.9751_weights.pth"
        self.classes = ('daisy','dandelion','rose','sunflower','tulip') 
        model = SELFMODEL(len(self.classes)) # 实例化模型对象
        model = torch.nn.DataParallel(model) # 数据并行化
        weights = torch.load(self.model_path) # 加载模型参数
        model.load_state_dict(weights) # 参数装载进模型的状态字典
        model.eval() # 设置为评估模式，可禁用梯度跟踪，加快推理
        model = model.to(device) # 模型移到设备（GPU）上
        self.model = model
        self.data_transforms = data_transforms_func(224)['val'] # 设置数据预处理方式
        self.initUI()
    
    # 界面初始化函数
    def initUI(self):
        font_title = QFont('楷体', 16) # 设置标题字体为楷体，大小为16
        font_main = QFont('楷体', 14) # 设置主要内容字体为楷体，大小为14
        img_detection_widget = QWidget() # 创建一个新的图片识别功能界面的主窗口部件
        img_detection_layout = QVBoxLayout() # 创建一个垂直布局，用于放置图片识别功能界面的子部件
        img_detection_title = QLabel("图片识别功能") # 创建一个标签，作为图片识别功能的标题，并设置字体
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget() # 创建一个中间部件，用于放置待识别的图片
        mid_img_layout = QHBoxLayout() # 创建一个水平布局，用于放置待识别的图片
        self.dec_img = QLabel() # 创建一个标签，用于显示待识别的图片
        self.dec_img.setAlignment(Qt.AlignCenter) # 设置待识别的图片的对齐方式为居中
        mid_img_layout.addWidget(self.dec_img) # 在水平布局中添加
        mid_img_widget.setLayout(mid_img_layout) # 设置中间部件的布局为之前创建的水平布局

        up_img_button = QPushButton("上传图片") # 创建一个上传图片按钮，文本为“上传图片”
        det_img_button = QPushButton("识别") # 创建一个识别图片按钮，文本为“识别”
        up_img_button.clicked.connect(self.upload_img) # 连接上传图片按钮的点击信号到上传图片的方法
        det_img_button.clicked.connect(self.detect_img) # 连接识别图片按钮的点击信号到识别图片的方法
        up_img_button.setFont(font_main) # 设置上传图片按钮的字体
        det_img_button.setFont(font_main) # 设置识别图片按钮的字体

        # 创建一个标签，用于显示识别状态的文字，初始状态为“等待识别”
        self.rrr = QLabel("等待上传：点击下方按钮上传花卉图片")  # 创建状态标签
        self.rrr.setFont(font_main)  # 设置状态标签的字体
        self.rrr.setAlignment(Qt.AlignCenter)

        # 设置上传图片按钮的样式，定义其文本颜色、悬停背景颜色、背景颜色、边框、圆角、内边距和外边距
        up_img_button.setStyleSheet("QPushButton{color:white}"  # 设置按钮文字颜色为白色
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"  # 设置按钮悬停时的背景颜色
                                    "QPushButton{background-color:rgb(48,124,208)}"  # 设置按钮的背景颜色
                                    "QPushButton{border:2px}"  # 设置按钮的边框宽度
                                    "QPushButton{border-radius:5px}"  # 设置按钮的圆角半径
                                    "QPushButton{padding:5px 5px}"  # 设置按钮的内边距
                                    "QPushButton{margin:5px 5px}")  # 设置按钮的外边距

        # 设置识别按钮的样式，定义其文本颜色、悬停背景颜色、背景颜色、边框、圆角、内边距和外边距
        det_img_button.setStyleSheet("QPushButton{color:white}"  # 设置按钮文字颜色为白色
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"  # 设置按钮悬停时的背景颜色
                                    "QPushButton{background-color:rgb(48,124,208)}"  # 设置按钮的背景颜色
                                    "QPushButton{border:2px}"  # 设置按钮的边框宽度
                                    "QPushButton{border-radius:5px}"  # 设置按钮的圆角半径
                                    "QPushButton{padding:5px 5px}"  # 设置按钮的内边距
                                    "QPushButton{margin:5px 5px}")  # 设置按钮的外边距

        # 将图片识别功能标题标签添加到垂直布局中，并设置其对齐方式为居中
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter) 
        # 将中间部件（包含左右图标签）添加到垂直布局中，并设置其对齐方式为居中
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter) 
        # 将状态标签添加到垂直布局中
        img_detection_layout.addWidget(self.rrr) 
        # 将上传图片按钮添加到垂直布局中
        img_detection_layout.addWidget(up_img_button)  # 添加上传图片按钮到布局
        # 将识别按钮添加到垂直布局中
        img_detection_layout.addWidget(det_img_button)  # 添加识别按钮到布局

        # 设置图片识别功能主窗口部件的布局为之前创建的垂直布局
        img_detection_widget.setLayout(img_detection_layout)  # 设置主窗口部件的布局

        # 设置标签页
        self.addTab(img_detection_widget, '花卉图片检测')
        self.setTabIcon(0, QIcon(r'data/flowers/daisy/5673551_01d1ea993e_n.jpg'))

    # 上传图片方法
    def upload_img(self):
        # 打开文件对话框，选择要上传的图片文件
        fileName,_ = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:  # 如果选择了文件
            # suffix = fileName.split(".")[-1]  # 获取文件的后缀名
            save_path = os.path.join("images/tmp", "tmp_upload")  # 设置保存路径
            if not os.path.isdir(save_path): # 如果路径不存在就创建
                os.makedirs(save_path)
                print("Save diretory {0:} is created".format(save_path))
            shutil.copy(fileName, save_path)  # 将选中的文件复制到保存路径
            # 读取图片文件并调整大小，然后统一显示
            im0 = cv2.imread(os.path.join(save_path,fileName))  # 读取图片文件
            resize_scale = self.output_size / im0.shape[0]  # 计算调整比例
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)  # 调整图片大小
            cv2.imwrite(r"images/tmp/upload_show_result.jpg", im0)  # 将调整后的图片保存
            self.img2predict = fileName  # 将文件路径赋值给img2predict
            self.origin_shape = (im0.shape[1], im0.shape[0])  # 保存原始图片的尺寸
            self.dec_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))  # 更新显示上传的图片
            self.rrr.setText("等待识别：点击下方识别按钮开始识别")  # 更新状态标签为“等待识别”

    # 调用模型进行识别
    def detect_img(self):
        source = self.img2predict  # 获取要检测的图片路径
        img = Image.open(source)  # 打开图片文件
        img = self.data_transforms(img)  # 对图片进行预处理
        img = img.unsqueeze(0)  # 扩展维度以适应模型输入
        img = img.to(device)  # 将图片数据转移到设备（如GPU）
        output = self.model(img)  # 使用模型进行预测
        
        _,label_id = torch.max(output,dim=1) # 获取预测结果的logits和标签ID
        predict_name = self.classes[label_id]  # 获取预测结果的类别名称
        self.rrr.setText("当前识别结果为：{0:}".format(predict_name))  # 更新状态标签为识别结果

    # 关闭事件 询问用户是否退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self,  # 弹出消息框，询问用户是否退出程序
                                    '毁灭程序',
                                    "真的要离开吗？（可怜）\n再多看看吗，好不容易做出来的。",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:  # 如果用户选择是
            reply2 = QMessageBox.question(self,  # 弹出消息框，询问用户是否退出程序
                                    '毁灭程序',
                                    "爱用不用！（嫌弃）",
                                    QMessageBox.Yes)
            if reply2 == QMessageBox.Yes: 
                self.close()  # 关闭程序
                event.accept()  # 接受关闭事件
        else:  # 如果用户选择否
            event.ignore()  # 忽略关闭事件

if __name__ == "__main__":  # 程序入口
    app = QApplication(sys.argv)  # 创建QApplication实例
    mainWindow = MainWindow()  # 创建主窗口实例
    mainWindow.show()  # 显示主窗口
    sys.exit(app.exec_())  # 进入主事件循环并执行
