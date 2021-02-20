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
from util import flow2rgb, AverageMeter,ArrayToTensor
from path import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', metavar='DIR', default='./Datasets/CREMI/orgCrop_xb_serial',
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
parser.add_argument('--multiscale-weights', '-w', default=[0.005, 0.01, 0.02, 0.08, 0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=True,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained',
                    default='./myData/11-07-23_28/flownets_bn,adam,400epochs,epochSize1000,b4,lr0.0001/checkpoint.pth.tar',
                    help='path to pre-trained model')

n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1e-3


def main():
    global args
    print(device)
    args = parser.parse_args()

    print('data loading . . . . . . . . . . . . . . . .')
    # Data loading
    input_transform = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.319, 0.319, 0.319], std=[0.249, 0.249, 0.249])
    ])
    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set) + len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # Label loading
    label_transform = transforms.Compose([
        ArrayToTensor(),
        # transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        # transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    label_pairs = []
    ext = 'png'
    label_dir = Path('G:/ISBI2020/Datasets/CREMI/labelCrop_xb_serial/')
    test_files = label_dir.files('*1.{}'.format(ext))
    for file in test_files:
        img_pair = file.parent / (file.namebase[:-1] + '2.{}'.format(ext))
        if img_pair.isfile():
            label_pairs.append([file, img_pair])
    print('{} samples found'.format(len(label_pairs)))

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    batch_time = AverageMeter()
    Losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    save_path = './test_netArch/72serial_result/serial72_test_SA/'
    for i, input in enumerate(val_loader):

        input = torch.cat(input, 1).to(device)
        timg, simg = input.chunk(2, dim=1)
        # compute output
        output = model(input)
        # compute loss
        loss_data, simg_numpy, transImg_numpy, target_numpy, def_grid = multiscaleLoss(input, output, weights=args.multiscale_weights)
        loss_smooth = alpha * get_smooth_loss(output, weights=args.multiscale_weights)
        loss = loss_data + loss_smooth

        # for (label1_file, label2_file) in tqdm(label_pairs):
        label1_file = label_pairs[i][0]
        label2_file = label_pairs[i][1]
        label1 = cv2.cvtColor(cv2.imread(label1_file, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], cv2.COLOR_GRAY2BGR)
        label2 = cv2.cvtColor(cv2.imread(label2_file, cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], cv2.COLOR_GRAY2BGR)

        label1 = label_transform(label1)
        label2 = label_transform(label2)
        label = torch.cat([label1, label2]).unsqueeze(0).to(device)
        tlabel, slabel = label.chunk(2, dim=1)
        translabel = F.grid_sample(slabel, def_grid)

        translabel_numpy = translabel[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        TSlabel = translabel_numpy.astype(np.uint8)
        # TSlabel = np.clip(translabel_numpy, 0, 1).astype(np.uint8)
        TSlabel_tosave = save_path + str(i) + '_stLabel.png'
        plt.imsave(TSlabel_tosave, TSlabel)
        tlabel_numpy = tlabel[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        # Tlabel = np.clip(tlabel_numpy, 0, 1) .astype(np.uint8)
        Tlabel = tlabel_numpy.astype(np.uint8)
        Tlabel_tosave = save_path + str(i) + '_tlabel.png'
        plt.imsave(Tlabel_tosave, Tlabel)
        slabel_numpy = slabel[0].transpose(0, 1).transpose(1, 2).detach().cpu().numpy()
        # Slabel = np.clip(slabel_numpy, 0, 1).astype(np.uint8)
        Slabel = slabel_numpy.astype(np.uint8)
        Slabel_tosave = save_path + str(i) + '_slabel.png'
        plt.imsave(Slabel_tosave, Slabel)

        if i % args.print_freq == 0:
            # save flow map
            suffix = '.png'
            div_flow = 20
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
            a = save_path  + str(i) + '_stImg.png'
            plt.imsave(a, TSimg)
            Simg = np.clip(simg_numpy, 0, 1)
            b = save_path  + str(i) + '_simg.png'
            plt.imsave(b, Simg)
            Timg = np.clip(target_numpy, 0, 1)
            c = save_path  + str(i) + '_timg.png'
            plt.imsave(c, Timg)

        # record EPE
        Losses.update(loss.item(), timg.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t Loss {3}'
                  .format(i, len(val_loader), batch_time, Losses))

    print(' * Loss {:.3f}'.format(Losses.avg))

    return Losses.avg


if __name__ == '__main__':
    main()