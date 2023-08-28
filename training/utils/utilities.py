import torch
import math
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr 
from skimage.metrics import structural_similarity as compare_ssim


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return PSNR/Img.shape[0]


def batch_SSIM(img, imclean, data_range):
    img = torch.transpose(img, 1, 3)
    imclean = torch.transpose(imclean, 1, 3)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range, multichannel=True)
    return SSIM/Img.shape[0]