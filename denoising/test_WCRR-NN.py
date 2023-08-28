import torch
from torch.utils.data import DataLoader
import argparse
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr 
import pandas as pd
sys.path.append('../')
from training.utils import utils
from training.data import dataset
from models import utils as model_utils
import time

torch.manual_seed(0)

def test(device, model, fname):

    test_dataset = dataset.H5PY("../training/data/preprocessed/BSD/test.h5")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    Sigma = [5, 15, 25]

    columns = ['sigma', 'image_id', 'psnr']
    df = pd.DataFrame(columns=columns)

    sigma = torch.tensor(Sigma).to(device).view(-1,1,1,1)


    for idx, im in enumerate(test_dataloader):
        im = im.to(device).repeat(len(Sigma),1,1,1)
        
        im_noisy = im + sigma/255*torch.empty_like(im).normal_()

        with torch.no_grad():
            im_denoised, _, _ = model_utils.accelerated_gd_batch(im_noisy, model, sigma=sigma, ada_restart=True, tol=1e-4)
        
        for i, sigma_n in enumerate(Sigma):
            psnr_ = psnr(im[0].cpu().numpy(), im_denoised[i].cpu().numpy(), data_range=1)
            df = pd.concat([df, pd.DataFrame([[sigma_n, idx, psnr_]], columns=columns)])
        print(f" **** {idx+1} **** Running averages:")
        print(df.groupby(['sigma'])['psnr'].mean())

    df.to_csv(f"test_WCRR_NN.csv")



if __name__ == "__main__":
     # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Testing WCRR-NN')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')

    args = parser.parse_args()

    # load model
    fname = 'WCRR-CNN'

    model = utils.load_model(fname, args.device)
    model.eval()
    model.to(args.device)
    # change the splines to sum of relus
    # this makes the computations faster
    # but it is done in a naive way => not always working/can yield a drop in performance
    #model.change_splines_to_clip()
    # high accuracy computation of \|W\|
    print(" **** Updating the Lipschitz constant **** ")
    sn_pm = model.conv_layer.spectral_norm(mode="power_method", n_steps=200)

    test(args.device, model, fname)