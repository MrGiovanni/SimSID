import os
from PIL import Image
import torch
from torchvision import transforms


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, img_size=(256, 256), normalize_tanh=False, positive_ratio=1.0):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        self.positive_ratio = positive_ratio
        print('positive ratio {:.2f}'.format(positive_ratio))

        # collect training/testing files
        if phase == 'train':
            with open(os.path.join(data_dir, 'train_squid.txt'), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(data_dir, 'train_squid_abnormal.txt'), 'r') as f:
                neg_lines = f.readlines()
            total = len(lines)
            num_pos = int(total * self.positive_ratio)
            num_neg = total - num_pos
            lines = lines[:num_pos]
            neg_lines = neg_lines[:num_neg]
            lines = lines + neg_lines
        elif phase == 'val':
            with open(os.path.join(data_dir, 'val_squid.txt'), 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(os.path.join(data_dir, 'test_COVIDx9A.txt'), 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                self.datalist.append((os.path.join(data_dir, 'train', fname), label))
            else:
                self.datalist.append((os.path.join(data_dir, 'test', fname), label))
        
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
        image = Image.open(fpath).convert('L')
        image = image.resize(self.img_size)
        image = self.transforms(image)
        label = 0 if label == 'normal' else 1
        label = torch.tensor([label], dtype=torch.long)
        return image, label