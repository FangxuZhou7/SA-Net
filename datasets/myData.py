import os
import os.path
import glob
from .listdataset import ListDataset
import numpy as np

def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def make_dataset(dir, split):
    images = []
    for name in glob.iglob(os.path.join(dir, '*_img1.png')):
        name = os.path.basename(name)
        root_filename = name[:-9]
        img1 = os.path.join(dir, root_filename+'_img1.png')
        img2 = os.path.join(dir, root_filename+'_img2.png')
        if not (os.path.isfile(os.path.join(dir, img1)) or os.path.isfile(os.path.join(dir, img2))):
            continue
        images.append([img1, img2])
    return split2list(images, split, default_split=0.8)

def myData(root, transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(root, train_list, transform)
    test_dataset = ListDataset(root, test_list, transform)

    return train_dataset, test_dataset






