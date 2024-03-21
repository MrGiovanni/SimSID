import os
import pandas as pd
from PIL import Image
import pydicom as dicom
import torch
from torchvision import transforms


class SIIM(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, img_size=(256, 256), normalize_tanh=False):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_squid.txt')) as f:
                fnames = f.readlines()
                fpaths = [os.path.join(data_dir, 'train', fname.strip()) for fname in fnames]
        elif phase == 'val':
            with open(os.path.join(data_dir, 'val_squid.txt')) as f:
                fnames = f.readlines()
                fpaths = [os.path.join(data_dir, 'train', fname.strip()) for fname in fnames]
        else:
            fnames = os.listdir(os.path.join(data_dir, 'test'))
            fpaths = [os.path.join(data_dir, 'test', fname.strip()) for fname in fnames]

        all_labels = pd.read_csv(os.path.join(data_dir, 'stage_2_train.csv'))
        fname2normal = dict()
        for i in all_labels.index:
            imageid = all_labels.loc[i, 'ImageId']
            enc_pixels = all_labels.loc[i, 'EncodedPixels']
            if enc_pixels == '-1':
                fname2normal[imageid] = 0
            else:
                fname2normal[imageid] = 1
        
        self.datalist = list()
        for i in range(len(fnames)):
            imageid = fnames[i].strip()[:-4]
            self.datalist.append((fpaths[i], fname2normal[imageid]))
        
        # transforms
        if phase == 'train':
            self.transforms = [
                transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95,1.05)),
                transforms.ToTensor()
            ]
        else:
            self.transforms = [transforms.ToTensor()]
        if normalize_tanh:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms = transforms.Compose(self.transforms)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        fpath, label = self.datalist[index]
        image = dicom.dcmread(fpath).pixel_array
        image = Image.fromarray(image).resize(self.img_size)
        image = self.transforms(image)
        if self.phase == 'train':
            assert label == 0
        label = torch.tensor([label], dtype=torch.long)
        return image, label
        