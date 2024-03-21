import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random


class Zhang(torch.utils.data.Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), normalize=False, normalize_tanh=False, enable_transform=True, full=True, positive_ratio=1.0):

        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.positive_ratio = positive_ratio

        if train:
            if enable_transform:
                self.transforms = [
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95,1.05)),
                    transforms.ToTensor()
                ]
            else:
                self.transforms = [transforms.ToTensor()]
        else:
            self.transforms = [transforms.ToTensor()]
        if normalize_tanh:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms = transforms.Compose(self.transforms)

        self.load_data()

    def load_data(self):
        self.fnames = list()
        if self.train:
            pos_items = os.listdir(os.path.join(self.root, 'NORMAL'))
            neg_items = os.listdir(os.path.join(self.root, 'PNEUMONIA'))
            total = len(pos_items)
            num_pos = int(total * self.positive_ratio)
            num_neg = total - num_pos
            for item in pos_items[:num_pos]:
                image = Image.open(os.path.join(self.root, 'NORMAL', item)).resize(self.img_size)
                self.data.append((image, 0))
                self.fnames.append(item)
            for item in neg_items[:num_neg]:
                image = Image.open(os.path.join(self.root, 'PNEUMONIA', item)).resize(self.img_size)
                self.data.append((image, 1))
                self.fnames.append(item)
        if not self.train:
            items = os.listdir(os.path.join(self.root, 'NORMAL'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'NORMAL', item)).resize(self.img_size), 0))
            self.fnames += items
            items = os.listdir(os.path.join(self.root, 'PNEUMONIA'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                self.data.append((Image.open(os.path.join(self.root, 'PNEUMONIA', item)).resize(self.img_size), 1))
            self.fnames += items
        print('%d data loaded from: %s, positive rate %.2f' % (len(self.data), self.root, self.positive_ratio))
    

    def __getitem__(self, index):
        img, label = self.data[index]

        img = self.transforms(img)[[0]]
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = Zhang('/media/administrator/1305D8BDB8D46DEE/jhu/ZhangLabData/CellData/chest_xray/val', train=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(trainloader):
        if img.shape[1] == 3:
            plt.imshow(img[0,1], cmap='gray')
            plt.show()
        break