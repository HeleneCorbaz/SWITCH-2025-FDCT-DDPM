'---- Calculating Metrics for Test in 3D from 2D predictions ----'

import nibabel as nib
import numpy as np
import os
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy.ndimage import zoom
from omegaconf import OmegaConf

config = OmegaConf.load('conf.yaml')

def resize_img(image, label):

    zooms = (label.shape[0]/image.shape[0], label.shape[1]/image.shape[1], label.shape[2]/image.shape[2])
    return zoom(image, zooms)

def calculate_metric(image, label):

    window_level = config.window_level
    window_width = config.window_width
    window_min = window_level - window_width/2
    window_max = window_level + window_width/2

    image = resize_img(image, label)
    label = np.array(label)
    label = np.clip(label, window_min, window_max)
    image = np.clip(image, window_min, window_max)

    data_range = window_width

    mse_value = mse(image, label)
    ssim_value = ssim(image, label, data_range=data_range)
    psnr_value = psnr(image, label, data_range=data_range)

    return mse_value, ssim_value, psnr_value

metrics_dict = {}

for f in os.listdir(config.inference.path_res):
    n = f.replace('.nii.gz', '')
    im = nib.load(os.path.join(config.inference.path_res, f))
    n = f.replace('FDCT', 'MDCT')
    im2 = nib.load(os.path.join(config.inference.path_mdct, n))
    arr = im.get_fdata()
    arr2 = im2.get_fdata()

    mse_values = []
    ssim_values = []
    psnr_values = []

    calculate_metric(arr, arr2)

    mse_value, ssim_value, psnr_value = calculate_metric(arr, arr2)

    mse_values.append(mse_value)
    ssim_values.append(ssim_value)
    psnr_values.append(psnr_value)

    metrics_dict[n] = {
        'mse': mse_values,
        'ssim': ssim_values,
        'psnr': psnr_values,
    }

    print(f"Metrics for {n}: {metrics_dict[n]}")

    with open(f'{n}.json', 'w') as json_file:
        json.dump(metrics_dict[n], json_file, indent=4)

print('Job done')
