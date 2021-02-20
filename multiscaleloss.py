
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def Loss(simg, input_flow, timg, mean=True):
    b, _, h, w = timg.size()

    # # create grid from -1 to 1
    x = np.arange(h).astype(np.float32)/h*2 - 1
    y = np.arange(w).astype(np.float32)/w*2 - 1
    xx, yy = np.meshgrid(x, y)
    xx_torch = torch.from_numpy(xx)
    xx_torch = xx_torch.unsqueeze(0).unsqueeze(1)
    yy_torch = torch.from_numpy(yy)
    yy_torch = yy_torch.unsqueeze(0).unsqueeze(1)
    grid = torch.cat((xx_torch, yy_torch), 1).repeat(b, 1, 1, 1).cuda()

    # generate deformable grid
    def_grid = torch.add(input_flow, grid)
    def_grid = def_grid.permute([0, 2, 3, 1])
    transImg = F.grid_sample(simg, def_grid)

    timg_crop = timg[:, :, 50:461, 50:461]
    transImg_crop = transImg[:, :, 50:461, 50:461]

    lossArch = SSIM()
    # loss_ssim = lossArch(timg, transImg) ##按1维度求2范数
    # loss_mse = torch.norm(timg - transImg, 2, 1)
    loss_ssim = lossArch(timg_crop, transImg_crop)
    loss_mse = torch.norm(timg_crop - transImg_crop, 2, 1)
    batch_size = loss_mse.size(0)

    if mean:
        return (loss_mse.mean() - loss_ssim.mean()*0.1), def_grid, transImg
    else:
        return loss_mse.sum()/batch_size

def realLoss(simage, output, timage):
    b, _, h, w = timage.size()
    upsampled_output = F.upsample(output, (h, w), mode='bilinear', align_corners=False)
    return Loss(simage, upsampled_output, timage, mean=True)

def multiscaleLoss(source, network_output, weights=None):

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_output))

    targetimg, sourceimg = source.chunk(2,dim =1)

    loss = 0
    for outputimg, weight in zip(network_output, weights):
        loss,def_grid,tImg =realLoss(sourceimg, outputimg, targetimg)
        loss =loss + weight * loss
    return loss, def_grid, tImg

def gradient(D):
    D_dy = D[:, :, 1:] - D[:, :, :-1]
    D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
    return D_dx, D_dy

def get_smooth_loss(network_outputs, weights=None):

    if type(network_outputs) not in [tuple, list]:
        network_outputs = [network_outputs]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    assert(len(weights) == len(network_outputs))
    smooth = 0
    for output, weight in zip(network_outputs, weights):

        output_dx, output_dy = gradient(output)
        x, y = output.chunk(2, dim=1)
        smooth1 = x.norm(2, 1).mean() + y.norm(2, 1).mean()
        smooth2 = output_dx.norm(2, 1).mean() + output_dy.norm(2, 1).mean()

        # smoothL1loss = torch.nn.L1Loss()
        # b_x, c_x, h_x, w_x = output_dx.shape
        # dx_zero = torch.zeros((b_x, c_x, h_x, w_x)).cuda()
        # b_y, c_y, h_y, w_y = output_dy.shape
        # dy_zero = torch.zeros((b_y, c_y, h_y, w_y)).cuda()
        # smooth2 = smoothL1loss(output_dx, dx_zero) + smoothL1loss(output_dy, dy_zero)

        smooth = smooth + weight * (smooth1+smooth2)

    return smooth





