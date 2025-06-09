import os
import numpy as np
import cv2

def load_images(folder):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')))
    images = []
    for idx, fname in enumerate(files):
        # 每隔5个加载一个文件
        if idx % 20 != 0:
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # convert grayscale to 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images


def images_to_pointcloud(images):
    pts = []
    cols = []
    for z, img in enumerate(images):
        h, w, _ = img.shape
        for i in range(h):
            for j in range(w):
                pts.append([i, j, z])
                cols.append(img[i, j, :])
    pts = np.vstack(pts)
    cols = np.vstack(cols)
    return pts, cols


def write_ply(filename, vertices, colors):
    n = vertices.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    with open(filename, 'w') as f:
        f.write("\n".join(header) + "\n")
        # 设置一个阈值，若 rgb 都小于该值则视为接近黑色，跳过写入
        threshold = 10
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            if r < threshold and g < threshold and b < threshold:
                continue
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main():
    # 将参数直接写在这里，不再从命令行读取
    input_folder = r"./CT"
    output_ply = r"./CT.ply"

    images = load_images(input_folder)
    if not images:
        print(f"No images found in folder: {input_folder}")
        return

    verts, cols = images_to_pointcloud(images)
    write_ply(output_ply, verts, cols)
    print(f"Saved point cloud with {verts.shape[0]} points to {output_ply}")


if __name__ == "__main__":
    main()
