import math
import sys
import os
import time
import torch
import torch.nn as nn

from models import basic_modules
from models.basic_modules import make_window, window_reverse
from models.memory import MemoryMatrixBlock, \
                   MemoryMatrixBlockV2, \
                   MemoryMatrixBlockV3, \
                   MemoryMatrixBlockV4
from models.inpaint import InpaintBlock


class PatchMemory(nn.Module):

    def __init__(self, in_channels, featmap_size, num_patches, num_slots):
        super().__init__()
        self.num_patches = num_patches
        self.num_slots = num_slots
        self.patch_size = featmap_size // num_patches
        self.slot_dim = 128 * self.patch_size ** 2
        self.memMatrix = nn.Parameter(torch.empty(self.num_patches ** 2, self.num_slots, self.slot_dim))
        self.qk_size = 256

        stdv = 1. / math.sqrt(self.memMatrix.size(-1))
        self.memMatrix.data.uniform_(-stdv, stdv)

        self.q_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.q_linear = nn.Linear(128 * self.patch_size ** 2, self.qk_size, bias=False)
        self.k_linear = nn.Linear(self.slot_dim, self.qk_size, bias=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        q = self.q_conv(x)
        batch_size, c, h, w = q.size()
        q = q.view(batch_size, c, self.num_patches, self.patch_size, self.num_patches, self.patch_size)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous()
        q = q.view(batch_size, self.num_patches ** 2, -1)
        q = self.q_linear(q)

        k = self.k_linear(self.memMatrix)
        scale = self.qk_size ** 0.5
        a = torch.matmul(q.unsqueeze(2), k.transpose(1, 2).unsqueeze(0)).squeeze(2) / scale    # shape (batch, patch ** 2, num_slots)
        a = torch.softmax(a, dim=-1)    # shape (batch, patch ** 2, num_slots)
        x = torch.matmul(a.unsqueeze(2), self.memMatrix.unsqueeze(0)).squeeze(2)    # shape (batch, patch ** 2, C)
        x = x.view(batch_size, self.num_patches, self.num_patches, c, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, c, h, w)
        x = self.out_conv(x)
        return x


class AE(nn.Module):
    def __init__(self, config, features_root, level=4):
        super(AE, self).__init__()
        self.config = config
        self.num_in_ch = config.num_in_ch
        self.initial_combine = config.initial_combine
        self.level = config.level
        self.ops = config.ops
        self.decoder_memory = config.decoder_memory
        print('SQUID ops:', config.ops)

        assert len(config.ops) == config.level

        self.filter_list = [features_root, features_root*2, features_root*4, features_root*8, features_root*16, features_root*16]

        self.in_conv = basic_modules.inconv(config.num_in_ch, features_root)
        
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(level):
            self.down_blocks.append(basic_modules.down(self.filter_list[i], self.filter_list[i+1], use_se=False))
            if config.ops[i] == 'concat':
                filter = self.filter_list[level-i] + self.filter_list[level-i-1]#//2
            else:
                filter = self.filter_list[level-i]
            self.up_blocks.append(basic_modules.up(filter, self.filter_list[level-1-i], op=config.ops[i], use_se=False))

        last_resolution = (config.img_size // config.num_patch) // (2 ** level)
        self.memory_blocks = nn.ModuleList()
        for i, mem_type in enumerate(config.decoder_memory):
            if mem_type == 'none':
                self.memory_blocks.append(nn.Identity())
            elif mem_type == 'V1':
                self.memory_blocks.append(
                    MemoryMatrixBlock(
                        config.memory_config,
                        self.filter_list[level-1-i] * (last_resolution * (2 ** (i + 1)) // int(math.sqrt(config.memory_config.num_memory))) ** 2
                    )
                )
            elif mem_type == 'V3':
                self.memory_blocks.append(
                    MemoryMatrixBlockV3(
                        config.memory_config,
                        self.filter_list[level-1-i] * (last_resolution * (2 ** (i + 1))) ** 2,
                        num_memory=config.num_patch ** 2,
                    )
                )
            elif mem_type == 'Patch':
                self.memory_blocks.append(
                    PatchMemory(
                        in_channels=self.filter_list[level-i],
                        featmap_size=config.img_size // 2 ** (level - i),
                        num_patches=2 ** (i + 1),
                        num_slots=20
                    )
                )

        self.out_conv = basic_modules.outconv(features_root, config.num_in_ch)
        if config.normalize_tanh:    # input is normalized to [-1, 1]
            self.out_nonlinear = nn.Tanh()
        else:
            self.out_nonlinear = nn.Sigmoid()

        self.mse_loss = nn.MSELoss()

        self.dist = config.dist
        if config.dist:
            self.teacher_ups = nn.ModuleList()
            for i in range(level):
                if config.ops[i] == 'concat':
                    filter = self.filter_list[level-i] + self.filter_list[level-i-1]
                else:
                    filter = self.filter_list[level-i]
                self.teacher_ups.append(basic_modules.up(filter, self.filter_list[level-1-i], op=config.ops[i], use_se=False))
            self.teacher_out = basic_modules.outconv(features_root, config.num_in_ch)

    def forward(self, x, fadein_weights=[1.0, 1.0, 1.0, 1.0]):
        """
        :param x: size [bs,C,H,W]
        :return:
        """
        x = self.in_conv(x)

        # encoding
        hiddens = list()
        for i in range(self.level):
            x = self.down_blocks[i](x)
            hiddens.append(x)
        
        t_x = x.clone().detach()
        
        after_memory = list()
        for i in range(len(self.decoder_memory)):
            m = self.memory_blocks[i](hiddens[-1-i])
            after_memory.append(m)

        # decoding
        decoded = list()
        x = after_memory[0]
        for i in range(self.level - 1):
            x = self.up_blocks[i](x, after_memory[i+1])
            decoded.append(x)
        x = self.up_blocks[-1](x)
        decoded.append(x)

        x = self.out_conv(x)
        x = self.out_nonlinear(x)

        if self.dist:
            t_decoded = list()
            for i in range(self.level - 1):
                t_x = self.teacher_ups[i](t_x, hiddens[-2-i].detach().clone())
                t_decoded.append(t_x)
            t_x = self.teacher_ups[-1](t_x)
            t_decoded.append(t_x)

            t_x = self.teacher_out(t_x)
            t_x = self.out_nonlinear(t_x)

            self_dist_loss = []
            for i in range(len(t_decoded)):
                self_dist_loss.append(self.mse_loss(decoded[i], t_decoded[i]))
            self_dist_loss = torch.sum(torch.stack(self_dist_loss))

        outs = dict(recon=x)
        if self.dist:
            outs['teacher_recon'] = t_x
            outs['dist_loss'] = self_dist_loss
        return outs
