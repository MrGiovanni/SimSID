import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np  


class SimpleDiscriminator(nn.Module):
    def __init__(self, num_in_ch=1, size=4, num_layers = 4, inplace=True):
        super(SimpleDiscriminator, self).__init__()

        self.size = size

        keep_stats = True

        out_channels = 16
        layers = [
            nn.Conv2d(num_in_ch, out_channels, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(16, track_running_stats=keep_stats), # this maybe required
            nn.LeakyReLU(0.2, inplace=inplace),
        ]
        
        for ilayer in range(num_layers):
            in_channels = out_channels
            out_channels = min(16 * (2 ** (ilayer + 1)), 256)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=True),
                nn.BatchNorm2d(out_channels, track_running_stats=keep_stats),
                nn.LeakyReLU(0.2, inplace=inplace),
            ])

        self.conv_model = nn.Sequential(*layers)

        self.regressor = nn.Linear(out_channels * size * size, 1)

    def forward(self, img):
        B = img.size(0)

        x = self.conv_model(img) # B, 128, W/16, H/16

        x = x.view(B, -1)
        x = self.regressor(x)
        return x
