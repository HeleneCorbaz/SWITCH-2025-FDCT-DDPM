'---- Dataloader for the inference ----'

import numpy as np
import monai
import nibabel as nib
import os
from omegaconf import OmegaConf

class SliceDataset(monai.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.config = OmegaConf.load('conf.yaml')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the 3D image and mask
        data = self.data[idx]
        image = nib.load(data["image"]).get_fdata().astype(np.float32)
        label = nib.load(data["label"]).get_fdata().astype(np.float32)

        window_level = self.config.window_level
        window_width = self.config.window_width

        # Apply windowing
        min_intensity = window_level - window_width // 2
        max_intensity = window_level + window_width // 2

        # Clip the data to the window range
        image = (image - min_intensity) / (max_intensity - min_intensity)
        label = (label - min_intensity) / (max_intensity - min_intensity)
        image = np.clip(image, 0, 1)
        label = np.clip(label, 0, 1)

        name = data['image']
        filename = os.path.basename(name)

        sample = self.transform({"image": image, "label": label, "filename": filename})

        return sample
