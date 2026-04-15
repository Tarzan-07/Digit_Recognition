"""
This file implements the safe batched loading of the dataset. 
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np
import os
import h5py
from PIL import Image

DATA = "data/"
TRAIN = "train/"
TEST = "test/"

# def constructFile(file: str):
#     if file == 'train':
#         return DATA+TRAIN
#     return DATA+TEST

class SVHN(Dataset):
    """
    Implementation of DataLoader for SVHN dataset. 
    """
    def __init__(self, split='train', transform=None):
        self.split = split
        self.file_path = DATA+TRAIN if split=='train' else DATA+TEST
        try:
            self.mat_path = os.path.join(self.file_path, 'digitStruct.mat')
        except Exception as e:
            print(f"digitStruct.mat file not found: {e}")

        self.file = h5py.File(self.mat_path, 'r')

        try:
            self.digit_name = self.file['digitStruct']['name']
            self.bbox = self.file['digitStruct']['bbox']
        except Exception as e:
            print(f"Relevant keys are missing in digitStruct.mat file: {e}")

        self.num_samples = self.digit_name.shape[0]
        self.transform = transform

    def _get_name(self, base_path, n):
        """
        Helper function to extract the file name of the n-th image. 
        """
        try:
            filename = f"{n}.png" or f"{n}.jpg"
        except Exception as e:
            print(f"Unable to open image. File name format should be n.png or n.jpg: {e}")

        return filename
    
    def _get_bbox(self, base_path, filename, n):
        bbox_ref = self.bbox[n][0]
        bbox_data = self.file[bbox_ref]

        def extract_attr(name):
            attr = bbox_data[name]

            if len(attr) > 1:
                return [self.file[attr[i][0]][0][0] for i in range(len(attr))]
            else:
                return attr[0][0]
            
        return {
            'label': extract_attr('label'),
            'top': extract_attr('top'),
            'left': extract_attr('left'),
            'width': extract_attr('width'),
            'height': extract_attr('height')
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        This function fetches a datapoint and loads it. This takes an indes as an input, and we
        need to return the corresponding image as the output. We can try some preprocessing
        in this function. 
        """

        img_name = self._get_name(index)
        boundary = self._get_bbox(index)

        img_path = self.file_path + f"{img_name}"
        image = Image.open(img_path)

        left = min(boundary['left'])
        top = min(boundary['top'])
        right = max([l + w for l, w in zip(boundary['left'], boundary['width'])])
        bottom = max([t + h for t, h in zip(boundary['top'], boundary['height'])])

        image = image.crop((left, top, right, bottom))
        image = image.resize((64, 64))

        labels = [int(l) if l != 10 else 0 for l in boundary['label']]

        target = torch.tensor(labels[0], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, target