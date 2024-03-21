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


class AE(nn.Module):
    def __init__(self, config, features_root, level=4):
        super(AE, self).__init__()
        self.config = config
        self.num_in_ch = config.num_in_ch
        self.initial_combine = config.initial_combine
        self.level = config.level
        self.num_patch = config.num_patch
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

        self.inpaint_block = InpaintBlock(
            config.inpaint_config,
            self.filter_list[level],
            num_memory=self.num_patch**2,
        )

        last_resolution = (config.img_size // config.num_patch) // (2 ** level)
        # self.memory_blocks = nn.ModuleList([
        #     nn.Identity(),
        #     nn.Identity(),
        #     MemoryMatrixBlock(config.memory_config, self.filter_list[level-2] * 4 * last_resolution ** 2) if config.ops[1] != 'none' else nn.Identity(),
        #     MemoryMatrixBlock(config.memory_config, self.filter_list[level-1] * last_resolution ** 2) if config.ops[0] != 'none' else nn.Identity(),
        # ])
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
        bs, C, H, W = x.size()
        assert W % self.num_patch == 0 or H % self.num_patch == 0

        squid_start_time = time.time()
        # segment patches
        x = make_window(x, window_size=W//self.num_patch, stride=W//self.num_patch, padding=0) # bs * num_patch**2, C, ws, ws
        num_windows = x.size(0) // bs

        x = self.in_conv(x)

        skips = []
        embeddings = list()
        # encoding
        for i in range(self.level):
            B_, c, w, h = x.size()
            if i < self.initial_combine:
                sc = window_reverse(x, w, h * self.num_patch, w * self.num_patch)
                skips.append(sc * fadein_weights[i])
            else:
                skips.append(x * fadein_weights[i])

            x = self.down_blocks[i](x)
            _, _, h, w = x.size()
            embedding = window_reverse(x, w, h * self.num_patch, w * self.num_patch)
            embeddings.append(embedding)

        encoder_time = time.time()
        B_, c, h, w = x.shape
        
        t_x = x.clone()
        if self.config.teacher_stop_gradient:
            t_x = t_x.detach()

        if self.config.use_memory_inpaint_block:
            x, alpha = self.inpaint_block(x, bs, num_windows, add_condition=True)
            embedding = window_reverse(x, w, h * self.num_patch, w * self.num_patch)
            embeddings.append(embedding)

        inpaint_time = time.time()

        self_dist_loss = []
        # decoding
        for i in range(self.level):
            # combine patches?
            if self.initial_combine is not None and self.initial_combine == (self.level - i):
                B_, c, h, w = x.shape
                x = window_reverse(x, w, h * self.num_patch, w * self.num_patch)
                t_x = window_reverse(t_x, w, h * self.num_patch, w * self.num_patch)
    
            x = self.up_blocks[i](x, skips[-1-i])
  
            # additional decoder memory matrix
            x = self.memory_blocks[i](x)
            if isinstance(x, tuple):    # memory block returns feature and attention weights
                x, _ = x

            if self.level - i > self.initial_combine:
                B_, c, h, w = x.shape
                embedding = window_reverse(x, w, h * self.num_patch, w * self.num_patch)
            else:
                embedding = x
            embeddings.append(embedding)

            if self.dist:
                t_x = self.teacher_ups[i](t_x, skips[-1-i].detach().clone())
                # do we need sg here? maybe not
                self_dist_loss.append(self.mse_loss(x, t_x))

        decode_time = time.time()

        # forward teacher decoder
        if self.dist:
            self_dist_loss = torch.sum(torch.stack(self_dist_loss))
            t_x = self.teacher_out(t_x)
            t_x = self.out_nonlinear(t_x)
            B_, c, w, h = t_x.shape
            if self.initial_combine is None:
                t_x = window_reverse(t_x, w, w * self.num_patch, w * self.num_patch)

        x = self.out_conv(x)
        x = self.out_nonlinear(x)

        out_time = time.time()
        # print('Encode time {}, inpaint time {}, decode time {}, out time {}'.format(
        #     encoder_time - squid_start_time, inpaint_time - encoder_time, decode_time - inpaint_time, out_time - decode_time
        # ))

        B_, c, w, h = x.shape

        if self.initial_combine is None:
            whole_recon = window_reverse(x, w, w * self.num_patch, w * self.num_patch)
        else:
            whole_recon = x
            x = make_window(x, W//self.num_patch, H//self.num_patch, 0)

        outs = dict(recon=whole_recon, embeddings=embeddings)
        if self.dist:
            outs['teacher_recon'] = t_x
            outs['dist_loss'] = self_dist_loss
        if self.config.analyze_memory:
            outs['alpha'] = alpha
        return outs
