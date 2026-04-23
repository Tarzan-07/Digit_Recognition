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
        elif self.split == 'eval':
            self.file_path = os.path.join(DATA, TEST)

        self.meta_data_path = os.path.join(self.file_path, 'digitStruct.mat')
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
    
    def get_cropped_digits(image_path: str, transform=None):
        """
        Load a single SVHN image, crop individual digits using metadata, and return list of (tensor, label) tuples.
        Assumes image_path is in 'data/test/' and filename is like '10669.png' (index starts at 1).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Extract image index
        img_name = os.path.basename(image_path)
        img_idx = int(os.path.splitext(img_name)[0]) - 1  # 0-based index
        
        # Load metadata
        meta_data_path = os.path.join(DATA, TEST, 'digitStruct.mat')
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(f"Metadata not found: {meta_data_path}")
        
        meta_data = h5py.File(meta_data_path, mode='r')
        bbox_ref = meta_data['digitStruct']['bbox'][img_idx][0]
        bbox_data = meta_data[bbox_ref]
        label_attr = bbox_data['label']
        num_digits = len(label_attr) if len(label_attr) > 1 else 1
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        cropped_digits = []
        
        for d_idx in range(num_digits):
            # Extract bbox (similar to _get_bbox_data)
            def get_val(attr_name):
                attr = bbox_data[attr_name]
                if len(attr) > 1:
                    ref = attr[d_idx][0]
                    return meta_data[ref][0][0]
                else:
                    return attr[0][0]
            
            info = {
                'label': int(get_val('label')) % 10,
                'top': int(get_val('top')),
                'left': int(get_val('left')),
                'width': int(get_val('width')),
                'height': int(get_val('height'))
            }
            
            # Crop
            crop_rect = (info['left'], info['top'], info['left'] + info['width'], info['top'] + info['height'])
            digit_image = image.crop(crop_rect)
            
            # Apply transform
            if transform:
                digit_image = transform(digit_image)
            else:
                digit_image = torch.from_numpy(np.array(digit_image)).permute(2, 0, 1).float() / 255.0
            
            cropped_digits.append((digit_image, info['label']))
        
        meta_data.close()
        return cropped_digits

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