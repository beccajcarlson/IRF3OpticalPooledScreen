import torch
from torch.utils.data import Dataset
from torchvision import transforms
import skimage
import skimage.io as io
import pandas as pd
import numpy as np

import os

class CellImageDataset46(Dataset):

    def __init__(self, datadir, metafile, mode='train', max_value = 65535, **kwargs):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.mode = mode
        self.max_value = max_value

        if mode == 'train':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])

        elif mode == 'val':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])

        else:
            raise KeyError("dataset mode must be one of ['train', 'val'], not %s" % mode)

    def __len__(self):
        return len(self.metadata)

    def load_image(self, fname, max_value):
        img = io.imread(fname, plugin='tifffile')
        
        refimg = io.imread(max_value, plugin = 'tifffile')
        from skimage.exposure import match_histograms        
        img = img * 255/max_value
        img[img > 255] = 255
        img = img.astype(np.uint8)
        img, _ = torch.max(torch.from_numpy(img).view(1, 1, 46, 46),1)
        return img

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]

        img = self.load_image(os.path.join(self.datadir, sample['filename']), self.max_value)
        img = self.transform(img)

        return {'image': img, 'key': sample['filename']}

class CellImageDataset(Dataset):

    def __init__(self, datadir, metafile, mode='train', **kwargs):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.mode = mode

        if mode == 'train':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
            ])

        elif mode == 'val':
            self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            ])

        else:
            raise KeyError("dataset mode must be one of ['train', 'val'], not %s" % mode)

    def __len__(self):
        return len(self.metadata)

    def load_image(self, fname):
        img = io.imread(fname, plugin='tifffile')
        img = img.astype(np.float32)
        img, _ = torch.max(torch.from_numpy(img).view(1, 1, 64, 64),1)
        print(img.shape)
        return img

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]

        img = self.load_image(os.path.join(self.datadir, sample['filename']))
        img = self.transform(img)

        return {'image': img, 'key': sample['filename']}

class CellImageDatasetwithTargets(CellImageDataset):
    def __init__(self, datadir, metafile, mode='train', target_names=['cell_type', 'eccentricity', 'roundness']):
        super(CellImageDatasetwithTargets, self).__init__(datadir=datadir, metafile=metafile, mode=mode)
        self.target_names = target_names

    def get_targets(self, sample):

        targets = []
        weights = []

        for n in self.target_names:
            targets.append(sample[n])
            weights.append(sample['weights'])

        return {'targets': torch.FloatTensor(targets), 'weights': torch.FloatTensor(weights)}

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]

        img = self.load_image(os.path.join(self.datadir, sample['filename']))
        img = self.transform(img)

        targets = self.get_targets(sample)

        return {'image': img, 'label': sample['label'], 'label_name': sample['label_name'], 'key': sample['filename'],
                'targets': targets['targets'], 'weights': targets['weights']}
