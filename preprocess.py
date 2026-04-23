import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import h5py

DATA = 'data'
TRAIN = 'train/train'
TEST = 'test'

class SVHNDataset(Dataset):
    """Implementation of Dataloader."""
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform

        if self.split == 'train':
            self.file_path = os.path.join(DATA, TRAIN)
            self.meta_data_path = os.path.join(self.file_path, 'digitStruct.mat')
        else:
            self.file_path = os.path.join(DATA, TEST)

        self.meta_data = h5py.File(name=self.meta_data_path, mode='r')

        try:
            self.digit_name = self.meta_data['digitStruct']['name']
            self.bbox = self.meta_data['digitStruct']['bbox']
            self.transform = transform
        except Exception as e:
            raise RuntimeError(f"Unable to parse the metadata file: {e}")
        
        self.digit_samples = []
        print(f"Indexing {self.split} digits.....this may take a moment")

        for i in range(len(self.digit_name)):
            bbox_ref = self.bbox[i][0]
            bbox_data = self.meta_data[bbox_ref]
            label_attr = bbox_data['label']

            num_digits = len(label_attr) if len(label_attr) > 1 else 1

            for d_idx in range(num_digits):
                self.digit_samples.append((i, d_idx))

        print(f"Found {len(self.digit_samples)} total digits.")

    def _get_bbox_data(self, img_idx, digit_idx):
        """Helper to extract specific digit data from HDF5."""
        bbox_ref = self.bbox[img_idx][0]
        bbox_data = self.meta_data[bbox_ref]

        def get_val(attr_name):
            attr = bbox_data[attr_name]
            if len(attr) > 1:
                ref = attr[digit_idx][0]
                return self.meta_data[ref][0][0]
            else:
                return attr[0][0]

        return {
            'label': int(get_val('label')) % 10,
            'top': int(get_val('top')),
            'left': int(get_val('left')),
            'width': int(get_val('width')),
            'height': int(get_val('height'))
        }

    def __len__(self):
        return len(self.digit_samples)

    def __getitem__(self, index):
        img_idx, digit_idx = self.digit_samples[index]
        info = self._get_bbox_data(img_idx, digit_idx)
        img_name = f"{img_idx + 1}.png"
        
        img_path = os.path.join(self.file_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        crop_rect = (
            info['left'], 
            info['top'], 
            info['left'] + info['width'], 
            info['top'] + info['height']
        )
        digit_image = image.crop(crop_rect)
        if self.transform:
            digit_image = self.transform(digit_image)
        else:
            digit_image = torch.from_numpy(np.array(digit_image)).permute(2, 0, 1).float() / 255.0

        return digit_image, info['label']

if __name__ == '__main__':
    dataset = SVHNDataset(split='train')
    img, label = dataset[0]
    print(f"Sample 0 - Label: {label}, Shape: {img.shape}")