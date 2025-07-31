'---- Training for image-to-image translation task from MONAI generative https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_ddpm/2d_ddpm_inpainting.ipynb'

import os
import time
import glob

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from monai import transforms
from slicewiseDS import SliceDataset
from monai.data import DataLoader
from monai.utils import set_determinism
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from skimage.metrics import structural_similarity as ssim

from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler

'--- Initial settings ---'
config = OmegaConf.load('conf.yaml')
writer = SummaryWriter()
torch.multiprocessing.set_sharing_strategy("file_system")
root_dir_mdct = config.path_mdct
root_dir_fdct = config.path_fdct

directory = os.environ.get("MONAI_DATA_DIRECTORY")
set_determinism(42)

train_images = sorted(glob.glob(os.path.join(root_dir_fdct, '*.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(root_dir_mdct, '*.nii.gz')))
data_dicts = [{"image": image_name, "label": label_name}
              for image_name, label_name in zip(train_images, train_labels)]
device = torch.device("cuda")

'--- Monai transforms ---'
train_transforms = transforms.Compose(
    [
        transforms.EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.Resized(keys=["image", "label"], spatial_size=(config.res, config.res),
              mode=("trilinear", "nearest")),
        transforms.ToTensor(),
    ]
)
'--- Data Loader ---'
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

print('-- Loading Training Data')
train_ds = SliceDataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
print("Number of training samples:", len(train_loader.dataset))

print('-- Loading Validation Data ')
val_ds = SliceDataset(data=val_files, transform=train_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

'--- Training ---'
print('-- Training --')
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
model.to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
inferer = DiffusionInferer(scheduler)

n_epochs = config.epochs

total_start = time.time()

def normalize(x):
    return x
    xmin = 0
    xmax = 100
    out = (x - xmin) / (xmax - xmin + 1e-8)
    out = out.clip(0, 1)
    return out

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    print('Starting epoch', epoch)

    for step, data in enumerate(train_loader):
        images = data["image"].to(device)
        seg = data["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)
        # Generate random noise
        noise = torch.randn_like(seg).to(device)
        noisy_seg = scheduler.add_noise(original_samples=seg, noise=noise, timesteps=timesteps)
        combined = torch.cat((images, noisy_seg), dim=1)
        prediction = model(x=combined, timesteps=timesteps)
            # Get model prediction
        loss = F.mse_loss(prediction.float(), noise.float())
        loss.backward()
        g = clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)
        writer.add_scalar(f'gradients', g.cpu().item(), epoch * len(train_loader) + step)
        # Print loss for the current iteration
        print(f"Epoch {epoch}, Step {step}, Loss Training: {loss.item()}")

    images = normalize(images)
    seg = normalize(seg)
    input_img = images
    noise = torch.randn_like(input_img).to(device)
    current_img = noise
    combined = torch.cat(
                    (input_img, noise), dim=1
                )

    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:
            with torch.no_grad():
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(
                                model_output, t, current_img
                            )
                combined = torch.cat((input_img, current_img), dim=1)
                current_img = normalize(current_img)
    writer.add_images('Training Inference - FDCT/MDCT/Prediction',
                                  torch.concat([images[0, 0, :, :], seg[0, 0, :, :], current_img[0, 0, :, :]], dim=0).clip(0, 1),
                                  epoch * len(train_loader) + step, dataformats='WH')
    model.eval()
    val_epoch_loss = 0
    val_epoch_ssim = 0

    for data_val in val_loader:
        images = data_val["image"].to(device)
        seg = data_val["label"].to(device)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)
        with torch.no_grad():
            noise = torch.randn_like(seg).to(device)
            noisy_seg = scheduler.add_noise(original_samples=seg, noise=noise, timesteps=timesteps)
            combined = torch.cat((images, noisy_seg), dim=1)
            prediction = model(x=combined, timesteps=timesteps)
            val_loss = F.mse_loss(prediction.float(), noise.float())
            val_epoch_loss += val_loss.item()

            prediction_np = prediction.detach().cpu().numpy()[0][0]
            noise_np = noise.detach().cpu().numpy()[0][0]  # Get the noise
            pred_min = prediction_np.min()
            pred_max = prediction_np.max()
            noise_min = noise_np.min()
            noise_max = noise_np.max()
            data_range = max(pred_max - pred_min, noise_max - noise_min)
            ssim_value = ssim(prediction_np, noise_np, data_range=data_range)
            val_epoch_ssim += ssim_value

            images = normalize(images)
            seg = normalize(seg)
            input_img = images

    print(f"Epoch {epoch}, Step {step}, Validation loss: {val_epoch_loss/len(val_loader)}")
    print(f"Epoch {epoch}, Step {step}, Validation ssim: {val_epoch_ssim/ len(val_loader)}")
    writer.add_scalar('Loss/validation', val_epoch_loss / len(val_loader), epoch)
    writer.add_scalar('SSIM/validation', val_epoch_ssim / len(val_loader), epoch)
    noise = torch.randn_like(input_img).to(device)
    current_img = noise
    combined = torch.cat((input_img, noise), dim=1)

    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:  # go through the noising process
        with torch.no_grad():
            model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
            current_img, _ = scheduler.step(model_output, t, current_img)
            combined = torch.cat((input_img, current_img), dim=1)
            current_img = normalize(current_img)
    writer.add_images('Validation Inference - FDCT/MDCT/Prediction',
                                          torch.concat([images[0, 0, :, :], seg[0, 0, :, :], current_img[0, 0, :, :]],
                                                       dim=0).clip(0, 1),
                                          epoch * len(train_loader) + step, dataformats='WH')

    torch.save(model.state_dict(), config.model_name)

