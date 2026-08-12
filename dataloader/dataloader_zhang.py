import os

import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
CLASS_DIRS = {
    'NORMAL': ('NORMAL', 'normal_256'),
    'PNEUMONIA': ('PNEUMONIA', 'pneumonia_256'),
}


def _candidate_roots(root):
    """Support both <split>/<class> and <split>/<split>/<class> layouts."""
    root = os.path.abspath(root)
    yield root

    split = os.path.basename(os.path.normpath(root))
    nested = os.path.join(root, split)
    if os.path.isdir(nested):
        yield nested


def _find_class_dir(root, class_name, required=True):
    for base in _candidate_roots(root):
        for dirname in CLASS_DIRS[class_name]:
            path = os.path.join(base, dirname)
            if os.path.isdir(path):
                return path

    if required:
        expected = ', '.join(CLASS_DIRS[class_name])
        raise FileNotFoundError(
            'Could not find {} data under {}. Expected one of: {}'.format(
                class_name, root, expected
            )
        )
    return None


def _list_images(path):
    if path is None:
        return []
    return sorted(
        name for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
        and name.lower().endswith(IMG_EXTS)
    )


class Zhang(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        img_size=(256, 256),
        normalize=False,
        normalize_tanh=False,
        enable_transform=True,
        full=True,
        positive_ratio=1.0,
    ):
        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.positive_ratio = positive_ratio

        if not 0.0 <= self.positive_ratio <= 1.0:
            raise ValueError('positive_ratio must be between 0 and 1')

        if train and enable_transform:
            transform_list = [
                transforms.RandomAffine(
                    0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                ),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [transforms.ToTensor()]

        if normalize_tanh:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        self.transforms = transforms.Compose(transform_list)
        self.load_data()

    def _append(self, directory, item, label):
        image = Image.open(os.path.join(directory, item)).resize(self.img_size)
        self.data.append((image, label))
        self.fnames.append(item)

    def load_data(self):
        self.fnames = []

        normal_dir = _find_class_dir(self.root, 'NORMAL', required=True)
        normal_items = _list_images(normal_dir)

        if self.train:
            total = len(normal_items)
            num_pos = int(total * self.positive_ratio)
            num_neg = total - num_pos

            pneumonia_dir = _find_class_dir(
                self.root, 'PNEUMONIA', required=(num_neg > 0)
            )
            pneumonia_items = _list_images(pneumonia_dir)

            if len(pneumonia_items) < num_neg:
                raise ValueError(
                    'Requested {} PNEUMONIA samples but only {} were found in {}'.format(
                        num_neg, len(pneumonia_items), pneumonia_dir
                    )
                )

            for item in normal_items[:num_pos]:
                self._append(normal_dir, item, 0)

            for item in pneumonia_items[:num_neg]:
                self._append(pneumonia_dir, item, 1)

        else:
            pneumonia_dir = _find_class_dir(
                self.root, 'PNEUMONIA', required=True
            )
            pneumonia_items = _list_images(pneumonia_dir)

            if not self.full:
                normal_items = normal_items[:10]
                pneumonia_items = pneumonia_items[:10]

            for item in normal_items:
                self._append(normal_dir, item, 0)

            for item in pneumonia_items:
                self._append(pneumonia_dir, item, 1)

        print(
            '%d data loaded from: %s, positive rate %.2f'
            % (len(self.data), self.root, self.positive_ratio)
        )

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
    dataset = Zhang(
        '/media/administrator/1305D8BDB8D46DEE/jhu/ZhangLabData/CellData/chest_xray/val',
        train=False,
    )
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0
    )

    for img, label in trainloader:
        if img.shape[1] == 3:
            plt.imshow(img[0, 1], cmap='gray')
        else:
            plt.imshow(img[0, 0], cmap='gray')
        plt.show()
        break
