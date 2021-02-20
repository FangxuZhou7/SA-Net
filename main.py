
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import datasets
import datetime
import argparse
import os
import time
from multiscaleloss import multiscaleLoss, get_smooth_loss
from util import EarlyStopping, AverageMeter, save_checkpoint, ArrayToTensor

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch SANet Training on EM image dataset',
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
group.add_argument('--split-value', default=0.9, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='SAnet_bn',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
# parser.add_argument('--div-flow', default=20,
#                     help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 1e-3
n_iter = 0


def main():
    global args
    print(device)
    args = parser.parse_args()

    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H_%M")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Dataset prepared
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
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # Create Model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Choose Optimizer
    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)



    # if args.evaluate:
    #     best_loss = validate(val_loader, model, 0)
    #     return
    print('__________________________')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    early_stop = EarlyStopping(patience=80, verbose=True)
    best_Loss = -1

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch)
        print(train_loss)
        scheduler.step()

        # evaluate on validation set
        with torch.no_grad():
            test_loss = validate(val_loader, model, epoch)

        early_stop(model, test_loss, epoch)
        if early_stop.early_stop:
            print("Early Stopping")
            break

        if best_Loss < 0:
            best_Loss = test_loss

        is_best = test_loss < best_Loss
        best_Loss = min(test_loss, best_Loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
        }, is_best, save_path)


def train(train_loader, model, optimizer, epoch):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
    end = time.time()

    for i, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = torch.cat(input, 1).to(device)
        optimizer.zero_grad()
        model.train()

        # compute output
        outputs = model(input)
        timg, simg = input.chunk(2, dim=1)
        loss_data, grid, transImg = multiscaleLoss(input, outputs, weights=args.multiscale_weights)

        input_reverse = torch.cat([simg, timg], dim=1).to(device)
        outputs_reverse = model(input_reverse)
        loss_data_re, grid_re, transImg_re = multiscaleLoss(input_reverse, outputs_reverse, weights=args.multiscale_weights)
        loss_consiss = torch.norm((grid+grid_re), 2).mean()

        loss_smooth = alpha * get_smooth_loss(outputs, weights=args.multiscale_weights)
        loss = loss_data + loss_smooth + 0.001 * loss_consiss
        losses.update(loss.item())

        # compute gradient and do optimization step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg


def validate(val_loader, model):
    global args

    batch_time = AverageMeter()
    Losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, input in enumerate(val_loader):
        input = torch.cat(input, 1).to(device)

        # compute output
        output = model(input)
        timg, simg = input.chunk(2, dim=1)
        loss_data, grid, transImg = multiscaleLoss(input, output, weights=args.multiscale_weights)
        loss_smooth = alpha * get_smooth_loss(output, weights=args.multiscale_weights)
        loss = loss_data + loss_smooth
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
