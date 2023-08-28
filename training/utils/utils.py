import json
import os
import glob
import sys
import torch
import matplotlib.pyplot as plt
sys.path.append('../')
from models.wc_conv_net import WCvxConvNet
from pathlib import Path

def load_model(name, device='cuda:0', epoch=None):
    # folder
    current_directory = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute()
    directory = f'{current_directory}/trained_models/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    rep = {"[": "[[]", "]": "[]]"}

    name_glob = name.replace("[","tr1u").replace("]","tr2u").replace("tr1u","[[]").replace("tr2u","[]]")

    # retrieve last checkpoint
    if epoch is None:
        files = glob.glob(f'{current_directory}/trained_models/{name_glob}/checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)


    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'
    # config file
    config = json.load(open(f'{directory}config.json'.replace("[[]","[").replace("[]]","]")))
   
    # build model

    model, _ = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    

    model.to(device)

    model.load_state_dict(checkpoint['state_dict'])
    model.conv_layer.spectral_norm()
    model.eval()

    return(model)

def build_model(config):
    # ensure consistency of the config file, e.g. number of channels, ranges + enforce constraints

    # 1- Activation function (learnable spline)
    param_spline_activation = config['spline_activation']
    # non expansive increasing splines
    param_spline_activation["slope_min"] = 0
    param_spline_activation["slope_max"] = 1
    # antisymmetric splines
    param_spline_activation["antisymmetric"] = True
    # shared spline
    param_spline_activation["num_activations"] = 1

    # 2- Multi convolution layer
    param_multi_conv = config['multi_convolution']
    if len(param_multi_conv['num_channels']) != (len(param_multi_conv['size_kernels']) + 1):
        raise ValueError("Number of channels specified is not compliant with number of kernel sizes")
    


    param_spline_scaling = config['spline_scaling']
    param_spline_scaling["clamp"] = False
    param_spline_scaling["x_min"] = config['noise_range'][0]
    param_spline_scaling["x_max"] = config['noise_range'][1]
    param_spline_scaling["num_activations"] = config['multi_convolution']['num_channels'][-1]


    model = WCvxConvNet(param_multi_conv=param_multi_conv, param_spline_activation=param_spline_activation, param_spline_scaling=param_spline_scaling, rho_wcvx=config['rho_wcvx'])


    return(model, config)
