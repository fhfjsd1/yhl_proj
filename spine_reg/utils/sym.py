# %%
import numpy as np
from PIL import Image

#!/usr/bin/env python3
# sym.py
# 读取图像，计算每列像素和，标注前5%最大列并绘图

import matplotlib.pyplot as plt

def main():

    # 1. 读取图像并转为灰度
    img = Image.open("/mnt/2097910f-f006-4c21-8b5a-0815530ed408/ICP/徐灏毕设代码/点云配准/unet/image.png").convert("L")
    arr = np.array(img)

    # 2. 计算每列像素和
    col_sums = arr.sum(axis=0)

    # 3. 找到前5%最大的列
    thresh = np.percentile(col_sums, 95)
    top5_idx = np.where(col_sums >= thresh)[0]

    # 4. 可视化：原图 + 列高亮
    fig, (ax_img, ax_sum) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_img.imshow(arr, cmap="gray")
    for x in top5_idx:
        ax_img.axvline(x, color="red", alpha=0.5, linewidth=1)
    ax_img.set_title("top5%")
    ax_img.axis("off")

    # 5. 绘制列和曲线
    ax_sum.plot(col_sums, label="列像素和")
    ax_sum.hlines(thresh, 0, len(col_sums)-1, colors="red", linestyles="--", label="95th 百分位")
    ax_sum.scatter(top5_idx, col_sums[top5_idx], color="red", s=10)
    ax_sum.set_title("每列像素和")
    ax_sum.set_xlabel("列索引")
    ax_sum.set_ylabel("像素和")
    ax_sum.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    