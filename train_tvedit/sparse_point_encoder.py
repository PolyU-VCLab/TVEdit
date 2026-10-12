# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self,
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding,
                bias=False,
                ):
        super().__init__()
        # possibly downsample at the first conv
        self.conv1 = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channel), 
                                  num_channels=out_channel)
        self.ac = nn.SiLU()
        # maintain the original shape at the second conv
        self.conv2 = nn.Conv2d(out_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=bias)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channel), 
                                  num_channels=out_channel)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.ac(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.ac(x)
        return x

class SparsePointEncoder(nn.Module):

    def __init__(
        self,
        input_dim=2,
        vae_downsample_scale=8,
        embed_dim=4,
        bias=False,
    ):
        super().__init__()
        self.handle_proj = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1,bias=False)
        self.target_proj = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1,bias=False)

        in_dim = embed_dim
        self.downsample_blocks_vae = nn.Sequential(*[
            ConvBlock((2**i)*in_dim, (2**(i+1))*in_dim,
            kernel_size=3, stride=2, padding=1, bias=bias)
            for i in range(int(np.log2(vae_downsample_scale*2)))
        ])
        out_dim = vae_downsample_scale*in_dim *2
        self.out = nn.Linear(out_dim,out_dim)

    def _zero_init(self):
        for linear in self.linears:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, handle_disk_map,target_disk_map,img=None):
        device = self.handle_proj.weight.device
        dtype = self.handle_proj.weight.dtype
        h = self.handle_proj(handle_disk_map.to(dtype).to(device))
        t = self.target_proj(target_disk_map.to(dtype).to(device))
        disk_map = torch.cat([h, t], dim=0) 

        embedding = self.downsample_blocks_vae(disk_map)
        embedding = embedding.flatten(2).transpose(1, 2)

        embedding = self.out(embedding)
        out= torch.chunk(embedding,chunks=2,dim=0)

        return out

if __name__=='__main__':
    p = PromptEncoder()
    x = torch.randn(1,1,512,512)
    y = torch.randn(1,1,512,512)
    z = p(x,y)
    import ipdb
    ipdb.set_trace()
    print(y)