from torchvision import datasets
from tqdm import tqdm

test_data = datasets.MNIST(root="./data/", train=False, download=True)


import os


saveDirTest = './DataImages-Test'


if not os.path.exists(saveDirTest):
    os.mkdir(saveDirTest)

def save_img(data, save_path):
    for i in tqdm(range(len(data))):
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-label-' + str(label) + '.png'))

save_img(test_data, saveDirTest)
