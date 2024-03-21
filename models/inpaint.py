import math
import sys
import os
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)
from models.basic_modules import *
from models import memory
import numpy as np


class InpaintBlock(nn.Module):
    def __init__(self, config, num_in_ch, num_memory=9):
        super(InpaintBlock, self).__init__()
        self.config = config
        self.num_in_ch = num_in_ch
        self.num_memory = num_memory
        self.mask_ratio = config.mask_ratio
        print('Masked Shortcut activated with mask ratio:', self.mask_ratio)
        memory_channel = config.memory_channel

        self.bottleneck_conv1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_in_ch // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(num_in_ch // 4),
            nn.ReLU(),
        )

        self.memory = nn.ModuleList()
        if config.use_memory_queue:
            for _ in range(num_memory):
                self.memory.append(memory.MemoryQueue(config.num_slots, memory_channel, shrink_thres=config.shrink_thres))
        else:
            for _ in range(num_memory):
                self.memory.append(memory.Memory(config.num_slots, memory_channel, config.shrink_thres))

        self.attention = nn.MultiheadAttention(memory_channel, 8)
        self.attn_norm1 = nn.LayerNorm(memory_channel)

        self.ff = nn.Sequential(
            nn.Linear(memory_channel, memory_channel // 4),
            nn.Dropout(config.drop),
            nn.ReLU(),
            nn.Linear(memory_channel // 4, memory_channel),
            nn.Dropout(config.drop),
        )
        self.attn_norm2 = nn.LayerNorm(memory_channel)
        
        self.bottleneck_conv2 = nn.Sequential(
            nn.Conv2d(num_in_ch // 4 + int(np.log2(num_memory)+1), num_in_ch, 1, 1, 0),
            # nn.Conv2d(num_in_ch // 4, num_in_ch, 1, 1, 0),
            nn.BatchNorm2d(num_in_ch),
            nn.ReLU()
        )

        self.binary_encoding = binarize(torch.arange(0,num_memory,dtype=torch.int), int(np.log2(num_memory)+1)) # num_memory, bits
   
        # mask to get the centre patch in a 3x3 neighborhood
        self.center_mask = torch.Tensor([True, True, True, True, False, True, True, True, True]).cuda().view(9,).bool()

        self.conv = nn.Conv2d(memory_channel, memory_channel, kernel_size=3)

    def forward(self, x, bs, num_windows, add_condition=False):
        """
        :param x: size [bs,C,H,W]
        :return:
        """
        B, OC, H, W = x.shape

        # bottleneck
        f_x = self.bottleneck_conv1(x)
        f_x = f_x.contiguous().view(bs, num_windows, -1)
        C = f_x.size(2)    # C = OC / 4 * H * W

        # space aware memory queue
        # TODO optimize
        n_x = torch.zeros_like(f_x)    # shape (bs, num_windows, C) or (bs, num_windows, OC // 4, H, W)
        alphas = list()
        for i in range(self.num_memory):
            # query
            m, alpha = self.memory[i](f_x[:, i])
            n_x[:,i] = n_x[:,i] + m
            alphas.append(alpha)
            # enque
            if self.config.use_memory_queue:
                self.memory[i].enque(f_x[:,i]) # enque only once at each step!
        alphas = torch.stack(alphas, axis=1)    # shape (bs, num_windows, num_slots)

        if self.config.use_inpaint:
            # formulate patches into 3x3 neighborhoods
            new_n_x = n_x.view(B, C, 1, 1)
            new_n_x = window_reverse(new_n_x, 1, int(num_windows**0.5),  int(num_windows**0.5)) # bs, C, sqrt(num_windows), sqrt(num_windows)
            new_n_x = make_window(new_n_x, 3, stride=1, padding=1)    # shape (B, C, 3, 3)
            new_n_x = new_n_x.permute(0, 2, 3, 1).view(B, -1, C) # B, 3*3, C
            # exclude the centre patch from 3x3 neighborhoods
            new_n_x = new_n_x[:, self.center_mask].permute(1, 0, 2).contiguous() # 8, B, C

            n_x = n_x.view(1, B, C) # N, B, C
            f_x = f_x.view(1, B, C) # N, B, C

            # transformer layer
            n_x = F.relu(self.attn_norm1(self.attention(f_x, new_n_x, new_n_x)[0]) + n_x)
            n_x = F.relu(self.attn_norm2(self.ff(n_x)) + n_x) # 1, B, C

            # for ablation
            # new_n_x[:, 4] = 0
            # new_n_x = new_n_x.view(B, 3, 3, C).permute(0, 3, 1, 2)
            # new_n_x = F.relu(self.conv(new_n_x))    # B, C, 1, 1
            # n_x = new_n_x
        
        n_x = n_x.view(bs, num_windows, OC // 4, H, W)
        f_x = n_x
        
        # add position condition, this may help?
        if add_condition:
            f_x = self.add_condition(f_x, bs, num_windows) # bs, num_windows, OC // 4, H, W

        f_x = f_x.view(bs*num_windows, -1, H, W)

        # bottleneck
        f_x = self.bottleneck_conv2(f_x) # B, OC, H, W

        if self.training:
            # spatial binary mask
            mask = torch.ones(f_x.size(0), 1, f_x.size(-2), f_x.size(-1)).to(f_x.device) * self.mask_ratio
            mask = torch.bernoulli(mask).float()

            f_x = mask * f_x + (1. - mask) * x

        return f_x, alphas

    # add binary position condition
    def add_condition(self, x, bs, num_windows):
        condition = self.binary_encoding.to(x.device).view(1, self.binary_encoding.shape[0], self.binary_encoding.shape[1], 1, 1)
        condition = condition.expand(bs, -1, -1, x.size(-2), x.size(-1)).contiguous().float()
        x = torch.cat((x, condition), dim=2)
        return x
