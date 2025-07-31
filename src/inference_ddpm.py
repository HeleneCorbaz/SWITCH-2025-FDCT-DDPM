import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from slicewiseDS_inference import SliceDataset
from monai.data import DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import json
from omegaconf import OmegaConf

from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler

def save_nifti(tensor, filename):
    image = tensor.cpu().detach().numpy()
    image = np.squeeze(image)
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nifti_image, filename)

def normalize(x):
    return x
    xmin = 0
    xmax = 100
    out = (x - xmin) / (xmax - xmin + 1e-8)
    out = out.clip(0, 1)
    return out

'--- Initial settings ---'
config = OmegaConf.load('conf.yaml')
root_dir_mdct = config.inference.path_mdct
root_dir_fdct = config.inference.path_fdct
path_val = config.inference.path_res
os.makedirs(path_val, exist_ok=True)
model_dict = config.inference.model_name

directory = os.environ.get("MONAI_DATA_DIRECTORY")

train_images = sorted(glob.glob(os.path.join(root_dir_fdct, '*.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(root_dir_mdct, '*.nii.gz')))

data_dicts = [{"image": image_name, "label": label_name}
              for image_name, label_name in zip(train_images, train_labels)]
device = torch.device("cuda")
j = 0

set_determinism(42)

'--- Monai transforms ---'
train_transforms = transforms.Compose(
    [
        transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Resized(keys=["image", "label"], spatial_size=(config.inference.res, config.inference.res),
              mode=("trilinear", "nearest")),
        transforms.ToTensor(),
    ]
)
'--- Data Loader ---'
val_ds = SliceDataset(data=data_dicts, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=config.inference.batch_size, shuffle=False)
print("Number of images to predict:", len(val_loader.dataset))

scheduler = DDPMScheduler(num_train_timesteps=1000)
inferer = DiffusionInferer(scheduler)
model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 128, 256, 256, 256, 256, 256, 256, 256),
    attention_levels=(False, False, False, False, False, False, False, False, True),
    num_res_blocks=4,
    num_head_channels=64,
    with_conditioning=False,
)
model.load_state_dict(torch.load(model_dict))
model.to(device)

val_total_loss = 0
val_total_ssim = 0
individual_mse = []
individual_ssim = []
j = 0

for data_val in val_loader:
    images = data_val["image"].to(device)
    seg = data_val["label"].to(device)
    name = data_val["filename"]
    timesteps = torch.randint(0, 1000, (len(images),)).to(device)

    with torch.no_grad():
        noise = torch.randn_like(seg).to(device)
        noisy_seg = scheduler.add_noise(original_samples=seg, noise=noise, timesteps=timesteps)
        combined = torch.cat((images, noisy_seg), dim=1)
        prediction = model(x=combined, timesteps=timesteps)
        val_loss = F.mse_loss(prediction.float(), noise.float())
        val_total_loss += val_loss.item()
        individual_mse.append(val_loss.item())

        prediction_np = prediction.detach().cpu().numpy()[0][0]
        noise_np = noise.detach().cpu().numpy()[0][0]
        pred_min = prediction_np.min()
        pred_max = prediction_np.max()
        noise_min = noise_np.min()
        noise_max = noise_np.max()
        data_range = max(pred_max - pred_min, noise_max - noise_min)
        ssim_value = ssim(prediction_np, noise_np, data_range=data_range)
        val_total_ssim += ssim_value
        individual_ssim.append(ssim_value)

        images = normalize(images)
        seg = normalize(seg)
        input_img = images

        noise = torch.randn_like(input_img).to(device)
        current_img = noise
        combined = torch.cat((input_img, noise), dim=1)
        scheduler.set_timesteps(num_inference_steps=1000)
        progress_bar = tqdm(scheduler.timesteps)

        for t in progress_bar:
            with torch.no_grad():
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(model_output, t,
                                                current_img)
                combined = torch.cat((input_img, current_img), dim=1)
                current_img = normalize(current_img)

        '-----Change ident to save last predicted image or all step images'
        for u in range(images.size(0)):
            prediction = current_img[u][0]
            filename = os.path.join(path_val, name[u])
            save_nifti(prediction, filename)

    j += images.size(0)

# Append values to the dictionary
individual_metrics = {}
for name, ssim_value, mse_value in zip(train_images, individual_ssim, individual_mse):
    individual_metrics[name] = {
        'SSIM': ssim_value,
        'MSE': mse_value
    }

json_file_path = os.path.join(path_val, 'individual_metrics.json')
with open(json_file_path, 'w') as json_file:
    json.dump(individual_metrics, json_file)

print('Job done')