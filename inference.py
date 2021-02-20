import argparse
import time
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
from multiscaleloss import multiscaleLoss, get_smooth_loss
from util import flow2rgb, AverageMeter, ArrayToTensor
import matplotlib.pyplot as plt
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', metavar='DIR', default='./data/mydata12',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='myData',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SAnet_bn',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=True,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default='./myData/10-28-14_47/flownets_bn,adam,400epochs,epochSize1000,b2,lr0.0001/checkpoint.pth.tar',
                    help='path to pre-trained model')



n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1e-3

def main():
    global args
    print(device)
    args = parser.parse_args()
    print('data loading . . . . . . . . . . . . . . . .')

    # Data
    input_transform = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.288, 0.288, 0.288], std=[0.323, 0.323, 0.323])
    ])
    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        split=args.split_file if args.split_file else args.split_value
    )

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))

    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0)
        return


def validate(val_loader, model, epoch):
    global args

    batch_time = AverageMeter()
    EPEs = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, input in enumerate(val_loader):

        input = torch.cat(input, 1).to(device)
        timg, simg = input.chunk(2, dim=1)
        # compute output
        output = model(input)
        # compute loss
        loss_data, simg_numpy, transImg_numpy, target_numpy,def_grid = multiscaleLoss(input, output, weights=args.multiscale_weights)
        loss_smooth = alpha * get_smooth_loss (output, weights=args.multiscale_weights)
        loss = loss_data + loss_smooth

        transImg = F.grid_sample(simg, def_grid)



        if i % args.print_freq == 0:
            save_path = 'D:/zhoufx/res/data8_CA6up432_test/'
            suffix = '.png'
            div_flow = 20

        # save flow map
            for n in range(5):
                outputshow = output[n]
                out = F.interpolate(outputshow, size=timg.size()[-2:], mode='bilinear', align_corners=False)
                filename = save_path + str(i) + '_' + str(pow(2, 7 - n)) + suffix
                oo = out[0]
                rgb_flow = flow2rgb(div_flow * oo, max_value=None)
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
                plt.imsave(filename, to_save)

        # save image
            TSimg = np.clip(transImg_numpy, 0, 1)
            a = save_path + str(i) + '_stImg.png'
            plt.imsave (a, TSimg)

            Simg = np.clip(simg_numpy, 0, 1)
            b = save_path + str(i) + '_simg.png'
            plt.imsave (b, Simg)

            Timg = np.clip(target_numpy, 0, 1)
            c = save_path + str (i) + '_timg.png'
            plt.imsave (c, Timg)

        EPEs.update(loss.item(), timg.size(0))
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, EPEs))

    print(' * EPE {:.3f}'.format(EPEs.avg))

    return EPEs.avg


if __name__ == '__main__':
    main()