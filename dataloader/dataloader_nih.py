import os
from collections import defaultdict
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms


class NIHChestXray(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, img_size=(256, 256), normalize_tanh=False, test_with_bbox=False):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        self.test_with_bbox = test_with_bbox

        if phase == 'train':
            with open(os.path.join(data_dir, 'train_squid.txt')) as f:
                fnames = f.readlines()
        elif phase == 'val':
            with open(os.path.join(data_dir, 'val_squid.txt')) as f:
                fnames = f.readlines()
        else:
            with open(os.path.join(data_dir, 'test_list.txt')) as f:
                fnames = f.readlines()
        self.fnames = [fname.strip() for fname in fnames]

        meta = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))
        self.labels = dict()
        for i in range(len(meta)):
            self.labels[meta.at[i, 'Image Index']] = meta.at[i, 'Finding Labels']
        
        bbox_info = pd.read_csv(os.path.join(data_dir, 'BBox_List_2017.csv'))
        self.bboxes = defaultdict(dict)
        for i in range(len(bbox_info)):
            x = bbox_info.at[i, 'Bbox [x']
            y = bbox_info.at[i, 'y']
            w = bbox_info.at[i, 'w']
            h = bbox_info.at[i, 'h]']
            self.bboxes[bbox_info.at[i, 'Image Index']][bbox_info.at[i, 'Finding Label']] = [x, y, w, h]

        if phase == 'test' and test_with_bbox:
            fnames_with_bbox = set()
            for i in range(len(bbox_info)):
                fnames_with_bbox.add(bbox_info.at[i, 'Image Index'])
            self.fnames = [fname for fname in self.fnames if fname in fnames_with_bbox]

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
        return len(self.fnames)
    
    def __getitem__(self, index):
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.data_dir, 'images', fname)).convert('L')
        ow, oh = image.size
        image = image.resize(self.img_size)
        image = self.transforms(image)
        label = 0 if self.labels[fname] == 'No Finding' else 1
        if self.phase == 'train':
            assert(label == 0)
        label = torch.tensor([label], dtype=torch.long)
        if self.phase == 'test' and self.test_with_bbox:
            bboxes = list()
            bbox_classes = list()
            for cls, [x, y, w, h] in self.bboxes[fname].items():
                bboxes.append([x, y, w, h])
                bbox_classes.append(cls)
            bboxes = torch.tensor(bboxes)
            factor_w = self.img_size[0] / ow
            factor_h = self.img_size[1] / oh
            bboxes[:, 0] = bboxes[:, 0] * factor_w
            bboxes[:, 2] = bboxes[:, 2] * factor_w
            bboxes[:, 1] = bboxes[:, 1] * factor_h
            bboxes[:, 3] = bboxes[:, 3] * factor_h
            return image, label, bboxes, bbox_classes
        else:
            return image, label
