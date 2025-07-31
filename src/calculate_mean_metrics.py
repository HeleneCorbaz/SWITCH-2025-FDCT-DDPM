'--- Calculate mean metrics for all cases ---'

import json
import os
import numpy as np

def calculate_means(json_files):
    mse_values = []
    ssim_values = []
    psnr_values = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            mse_values.append(data['mse'][0])
            ssim_values.append(data['ssim'][0])
            psnr_values.append(data['psnr'][0])

    mean_mse = np.mean(mse_values)
    mean_ssim = np.mean(ssim_values)
    mean_psnr = np.mean(psnr_values)

    std_mse = np.std(mse_values)/len(json_files)
    std_ssim = np.std(ssim_values)/len(json_files)
    std_pnsr = np.std(psnr_values)/len(json_files)

    return {
        'mean_mse': mean_mse,
        'mean_ssim': mean_ssim,
        'mean_psnr': mean_psnr,
        'std_mse': std_mse,
        'std_ssim': std_ssim,
        'std_psnr': std_pnsr,


    }

directory = 'path/to/your/project/'
json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

means = calculate_means(json_files)

output_file = 'mean_values.json'
with open(output_file, 'w') as f:
    json.dump(means, f, indent=4)

print('Job done')