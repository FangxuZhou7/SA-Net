
import os
import shutil
import torch
import numpy as np

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))  # handle numpy array
        tensor = torch.from_numpy(array)  # put it from HWC to CHW format
        return tensor.float()

class EarlyStopping(object):
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.count = 0
        self.best_loss = None
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, model, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            # self.save_checkpoint(val_loss, model, epoch)
        elif val_loss > self.best_loss:
            self.count += 1
            if self.verbose:
                print('EarlyStopping counter:\
                     {} out of {}'.format(self.count, self.patience))
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.count = 0

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    normalized_rgb_map = rgb_map / np.abs(rgb_map).max()
    return normalized_rgb_map  #rgb_map.clip(0,1)


# def get_mean_std(dataset, ratio=0.01):
#         """Get mean and std by sample ratio
#         """
#         dataloader = torch.utils.data.DataLoader (dataset, batch_size=int (len (dataset) * ratio),
#                                                   shuffle=True, num_workers=10)
#         train = iter (dataloader).next ()[0]  # 一个batch的数据
#         mean = np.mean (train.numpy(), axis=(0, 2, 3))
#         std = np.std (train.numpy (), axis=(0, 2, 3))
#         return mean, std
