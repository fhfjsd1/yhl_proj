import os
import torchvision.datasets as datasets

_DATASETS_MAIN_PATH = r'data'
_dataset_path = {
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
}

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'mnist':
        return datasets.MNIST(root=_dataset_path['mnist'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
