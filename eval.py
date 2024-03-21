import torch
torch.set_printoptions(10)

import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import shutil
from matplotlib import pyplot as plt

from models.squid import AE

import random
import argparse

import importlib

from tools import parse_args, build_disc, log, log_loss, save_image
from alert import GanAlert

import time

args = parse_args()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    print('exp folder cannot be found!')
    exit()

if not os.path.isfile(os.path.join('checkpoints', args.exp, 'discriminator.pth')):
    print('discriminator ckpt cannot be found!')
    exit()

if not os.path.isfile(os.path.join('checkpoints', args.exp, 'config.py')):
    print('config file cannot be found!')
    exit()

# load config file from exp folder
CONFIG = importlib.import_module('checkpoints.'+args.exp+'.config').Config()

save_path = os.path.join('checkpoints', args.exp, 'test_images')

# log
log_file = open(os.path.join('checkpoints', args.exp, 'eval_log.txt'), 'w')

# build main model from exp folder
MODULE = importlib.import_module('checkpoints.'+args.exp+'.squid')
model = MODULE.AE(CONFIG, 32, level=CONFIG.level).cuda()

print('Loading AE...')
ckpt = torch.load(os.path.join('checkpoints',args.exp,'model.pth'))
model.load_state_dict(ckpt)
print('AE loaded!')

# for discriminator
discriminator = build_disc(CONFIG).cuda()

print('Loading discriminator...')
ckpt = torch.load(os.path.join('checkpoints',args.exp,'discriminator.pth'))
discriminator.load_state_dict(ckpt)
print('discriminator loaded!')

# alert
alert = GanAlert(discriminator=discriminator, args=args, CONFIG=CONFIG, generator=model)

# from dataloader.dataloader_chexpert import CheXpert
# from dataloader.dataloader_zhang import Zhang
# # test_dataset = CheXpert(
# #     '/mnt/data0/yixiao/chexpert'+'/our_test_256_'+'pa', 
# #     train=False, 
# #     img_size=(128, 128),
# #     normalize_tanh=True,
# #     full=True,  
# #     data_type='pa',
# #     test_disease_type='all',
# #     positive_ratio=1.0,
# # )
# test_dataset = Zhang(
#     '/mnt/data0/yixiao/zhanglab-chest-xrays/resized256'+'/CellData/chest_xray/test',
#     train=False,
#     img_size=(128, 128),
#     normalize_tanh=True,
#     full=True
# )
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)


def evaluation():
    start_time = time.time()
    reconstructed, inputs, scores, labels = test(CONFIG.test_loader)
    print(time.time() - start_time)
    results = alert.evaluate(scores, labels, collect=True)

    # log metrics
    msg = '[TEST metrics] '
    for k, v in results.items():
        if np.isscalar(v):
            msg += k + ': '
            msg += '%.2f ' % v
    log(log_file, msg)
    print(msg)
    for f, t in zip(results['fpr'], results['tpr']):
        log_file.write(f'{f}_{t}\n')
    with open(os.path.join('checkpoints', args.exp, 'prcurve_log.txt'), 'w') as f:
        for p, r in zip(results['precisions'], results['recalls']):
            f.write(f'{p}_{r}\n')

    save_image(os.path.join(save_path, 'test'), zip(reconstructed, inputs))

def test(dataloader):
    model.eval()

    # for reconstructed img
    reconstructed = []
    # for input img
    inputs = []
    # for anomaly score
    scores = []
    # for gt labels
    labels = []

    count = 0
    for i, (img, label) in enumerate(dataloader):
        count += img.shape[0]
        img = img.to(CONFIG.device)
        label = label.cpu()

        out = model(img)
        fake_v = discriminator(out['recon'])

        scores += list(fake_v.detach().cpu().numpy())
        labels += list(label.detach().cpu().numpy())
        recon = out['recon'].detach().cpu().numpy()
        input = img.detach().cpu().numpy()

        if CONFIG.normalize_tanh:
            recon = ((recon + 1) / 2 * 255).astype(np.uint8)
            input = ((input + 1) / 2 * 255).astype(np.uint8)
        else:
            recon = (recon * 255).astype(np.uint8)
            input = (input * 255).astype(np.uint8)

        reconstructed += list(recon)
        inputs += list(input)

    return reconstructed, inputs, scores, labels


if __name__ == '__main__':
    evaluation()
