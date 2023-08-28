import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import sys

sys.path.append('/home/goujon/weakly_convex_ridge_regularizer/')
from training.utils import utils
from training.data import dataset

test_dataset = dataset.BSD500("/home/goujon/weakly_convex_ridge_regularizer/training/data/preprocessed/test_BSD.h5", shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

Sigma = [5, 10, 15, 20, 25, 30]

torch.manual_seed(42)
for sigma in Sigma:
    for idx, im in enumerate(test_dataloader):
        
    
        im_noisy = im + sigma/255*torch.empty_like(im).normal_()
        np.save(f'./{sigma}/test_noisy_{idx + 1}.npy', im_noisy[0,0].numpy())
        np.save(f'./{sigma}/test_clean_{idx + 1}.npy', im[0,0].numpy())




