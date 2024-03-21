import os
from PIL import Image
import torch
from torchvision import transforms


class Mvtec(torch.utils.data.Dataset):
    def __init__(self, data_dir, class_name, phase, img_size=(256, 256), normalize_tanh=False):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size

        self.datalist = list()
        all_classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                       'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                       'transistor', 'wood', 'zipper']
        if phase == 'train':
            assert class_name in all_classes
            for fname in os.listdir(os.path.join(data_dir, class_name, 'train/good')):
                self.datalist.append((os.path.join(data_dir, class_name, 'train/good', fname), 0))
        else:
            assert class_name in all_classes
            for subdir in os.listdir(os.path.join(data_dir, class_name, 'test')):
                if subdir == 'good':
                    for fname in os.listdir(os.path.join(data_dir, class_name, 'test', subdir)):
                        self.datalist.append((os.path.join(data_dir, class_name, 'test', subdir, fname), 0))
                else:
                    for fname in os.listdir(os.path.join(data_dir, class_name, 'test', subdir)):
                        self.datalist.append((os.path.join(data_dir, class_name, 'test', subdir, fname), 1))
        
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
        image = Image.open(fpath).convert('RGB')
        image = image.resize(self.img_size)
        image = self.transforms(image)
        label = torch.tensor([label], dtype=torch.long)
        return image, label
