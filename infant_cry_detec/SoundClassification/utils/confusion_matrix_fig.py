# #!/usr/bin/env python3
# """Helper to create Confusion Matrix figure

# Authors
#  * David Whipps 2021
#  * Ala Eddine Limame 2021
# """

# import itertools

# import matplotlib.pyplot as plt
# import numpy as np


# def create_cm_fig(cm, display_labels):
#     fig = plt.figure(figsize=cm.shape, dpi=50, facecolor="w", edgecolor="k")
#     ax = fig.add_subplot(1, 1, 1)

#     ax.imshow(cm, cmap="Oranges")  # fits with the tensorboard colour scheme

#     tick_marks = np.arange(cm.shape[0])

#     ax.set_xlabel("Predicted class", fontsize=18)
#     ax.set_xticks(tick_marks)
#     ax.set_xticklabels(display_labels, ha="center", fontsize=18, rotation=90)
#     ax.xaxis.set_label_position("bottom")
#     ax.xaxis.tick_bottom()

#     ax.set_ylabel("True class", fontsize=18)
#     ax.set_yticks(tick_marks)
#     ax.set_yticklabels(display_labels, va="center", fontsize=18)
#     ax.yaxis.set_label_position("left")
#     ax.yaxis.tick_left()

#     fmt = "d"  # TODO use '.3f' if normalized
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         ax.text(
#             j,
#             i,
#             format(cm[i, j], fmt),
#             horizontalalignment="center",
#             verticalalignment="center",
#             color="white" if cm[i, j] > thresh else "black",
#             fontsize=18,
#         )

#     fig.set_tight_layout(True)
#     # 保存图像
#     output_path = 'confusion_matrix.png'
#     fig.savefig(output_path)
#     return fig

# import itertools
# import matplotlib.pyplot as plt
# import numpy as np

# def create_cm_fig(cm, display_labels):
#     fig = plt.figure(figsize=cm.shape, dpi=50, facecolor="w", edgecolor="k")
#     ax = fig.add_subplot(1, 1, 1)

#     # 绘制混淆矩阵
#     cax = ax.imshow(cm, cmap="Oranges")  # fits with the tensorboard colour scheme

#     # 添加颜色条
#     fig.colorbar(cax)

#     tick_marks = np.arange(cm.shape[0])

#     # 设置 x 轴
#     ax.set_xlabel("Predicted class", fontsize=18)
#     ax.set_xticks(tick_marks)
#     ax.set_xticklabels(display_labels, ha="center", fontsize=18, rotation=90)
#     ax.xaxis.set_label_position("bottom")
#     ax.xaxis.tick_bottom()

#     # 设置 y 轴
#     ax.set_ylabel("True class", fontsize=18)
#     ax.set_yticks(tick_marks)
#     ax.set_yticklabels(display_labels, va="center", fontsize=18)
#     ax.yaxis.set_label_position("left")
#     ax.yaxis.tick_left()

#     # 在每个位置显示值
#     fmt = 'd'  # 格式化字符串，整数
#     thresh = cm.max() / 2.  # 阈值，用于设置文本颜色
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         ax.text(j, i, format(cm[i, j], fmt),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")

#     # 保存图像
#     output_path = 'confusion_matrix.png'
#     fig.savefig(output_path)
    
#     return fig

import matplotlib.pyplot as plt
import numpy as np

def create_cm_fig(cm, display_labels):
    """
    绘制混淆矩阵并保存图像。

    参数:
    cm (numpy.ndarray): 混淆矩阵数据。
    display_labels (list): 类别标签。
    output_path (str): 保存图像的路径。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制混淆矩阵
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    
    # 添加颜色条
    plt.colorbar(cax)
    
    # 设置 x 轴和 y 轴
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    
    # 在每个位置显示值
    fmt = 'd'  # 格式化字符串，整数
    thresh = cm.max() / 2.  # 阈值，用于设置文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 设置标签
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    return fig