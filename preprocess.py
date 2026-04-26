import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import SVHN
import os
import h5py
from scipy.io import loadmat
from PIL import Image

DATA_DIR = "./data"
TRAIN_FILE = os.path.join(DATA_DIR, "TRAIN", "train_32x32.mat")
TEST_FILE = os.path.join(DATA_DIR, "TEST", "test_32x32.mat")


class SVHNDataset(Dataset):
    def __init__(self, split, transform=None):
        # super().__init__()
        self.split = split
        self.transform = transform
        self.root_path = TRAIN_FILE if self.split == 'train' else TEST_FILE
        self.data = loadmat(self.root_path)
        self.images = self.data['X']
        self.labels = self.data['y'].squeeze().astype(int)
        self.labels[self.labels == 10] = 0

    def __len__(self):
        return self.images.shape[3]

    def __getitem__(self, index):

        img = self.images[:, :, :, index]
        label = int(self.labels[index])

        # img = torch.from_numpy(img).permute(2, 0, 1).float()/255.0

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label  
        # return image, label

def main():
    s = SVHNDataset(split='train')
    print(s.__getitem__(5))

if __name__ == '__main__':
    main()