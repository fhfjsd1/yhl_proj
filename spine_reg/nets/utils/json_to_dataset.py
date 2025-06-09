
import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

'''
labelme版本是3.16.7，
此处生成的标签图是8位彩色图，每个像素点的值就是这个像素点所属的种类

'''
if __name__ == '__main__':
    # 输入路径（图片和json文件所在路径）
    origin_path = "mydata/sourceImages"

    # 输出路径
    # TODO 自动创建文件夹
    jpgs_path = "mydata/JPEGImages"
    pngs_path = "mydata/SegmentationClass"  # 分割后的图片所在文件夹
    # 标签（输出成图像）
    # classes = ["_background_","cat","dog"]
    output_label_list = ["_background_", "bone", "zhongxian"]  # 添加需要用上的标签，第一个必须是_background_

    count = os.listdir(origin_path)
    for i in range(0, len(count)):
        path = os.path.join(origin_path, count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
            for label_name in label_names:
                index_json = label_names.index(label_name)
                try:
                    index_all = output_label_list.index(label_name)
                    new = new + index_all * (np.array(lbl) == index_json)   # 该处理与画图有关
                except:
                    print("\033[1;31m" + "标签 " + label_name + " 被忽略" + "\033[0m")   # 红色高亮显示被忽略的label

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
