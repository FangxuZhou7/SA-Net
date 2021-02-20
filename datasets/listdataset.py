import torch.utils.data as data
import os
import os.path

import numpy as np
import cv2

def default_loader(root, path_imgs):
    imgs = [os.path.join(root, path) for path in path_imgs]
    a = [cv2.cvtColor(cv2.imread(img, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], cv2.COLOR_GRAY2BGR).astype(np.float32) for img in imgs]
    return a

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        inputs = self.path_list[index]
        inputs = self.loader(self.root, inputs)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        return inputs

    def __len__(self):
        return len(self.path_list)

