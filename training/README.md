## Data Preparation
(mostly taken from [link](https://github.com/uclaopt/Provable_Plug_and_Play/tree/master/training)).

The BSD images are provided in the data folder.

To generate the preprocessed ```.h5``` data set files (data augmentation + patch extraction for training):

```console
~/weakly_convex_ridge_regularizers/training/data$ python BSD_preprocessing.py
```

## Training
To launch the training on, e.g., GPU \#0:
```console
~/weakly_convex_ridge_regularizers/training$ python train.py --device cuda:0
```

Follow the training with tensorboard:
```console
~/weakly_convex_ridge_regularizers/trained_models$ tensorboard --logdir .
```
The training parameters can be accessed/modified in ```/configs/config.json```.
 Default parameters:
 ```json
 {
    "exp_name": "WCRR-CNN",//name for saving the model
    "logging_info": {
        "log_batch": 500,// period of validation
        "log_dir": "../trained_models/"//output dir
    },
    "multi_convolution": {
        "num_channels": [// channels for the multiconvolution layer
            1,
            4,
            8,
            60
        ],
        "size_kernels": [// kernel sizes for the multiconvolution layer
            5,
            5,
            5
        ]
    },
    "noise_range": [// noise-levels range for training
        0,
        30
    ],
    "rho_wcvx": 1,// weak-convextiy bound
    "noise_val": 25,// noise-level for validation
    "optimization": {
        "lr": {// learning rates
            "conv": 0.005,
            "spline_activation": 5e-05,
            "mu": 0.05,
            "spline_scaling": 0.005
        }
        
    },
    "spline_activation": {// linear spline corresponding to the gradient of the potential function
        "num_knots": 101,// number of breakpoints
        "x_max": 0.1,// right-most knot
        "x_min": -0.1// left-most knot
    },
    "spline_scaling": {// linear spline for learning alpha(sigma)
        "init": 5.0,
        "num_knots": 11
    },
    "train_dataloader": {
        "batch_size": 128,
        "num_workers": 1,
        "train_data_file": "data/preprocessed/BSD/train.h5"
    },
    "training_options": {
        "fixed_point_solver_bw_params": {// computing gradient wrt fixed point with Anderson from DEQ framewrok
            "max_iter": 50,
            "tol": 0.001
        },
        "fixed_point_solver_fw_params": {// forward pass with AGD
            "max_iter": 200,
            "tol": 0.0001
        },
        "n_batches": 6000,// stop training condition
        "scheduler": {
            "gamma": 0.75,
            "n_batch": 500,
            "nb_steps": 10,
            "use": true
        }
    },
    "val_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "val_data_file": "data/preprocessed/BSD/validation.h5"
    }
}
```