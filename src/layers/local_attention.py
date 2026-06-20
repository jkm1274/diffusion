import torch
import torch.nn as nn
import torch.nn.functional as F
import src.layers.masks as sparse
import numpy as np
from einops import repeat


class LocalAttn(nn.Module):
    def __init__(self, in_channels, shrink=0, masks=None, nH=1):
        super(LocalAttn, self).__init__()
        
        self.in_channels = in_channels
        self.conv1x1_theta = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv1x1_phi = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv1x1_g = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv1x1_attn = nn.Conv1d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))
        self.shrink = shrink
        self.masks = masks
        self.nH = nH # should % 4 == 0

    def forward(self, x):
        batch_size, num_channels, h = x.size()
        location_num = h
        downsampled_num = location_num // 2
        hidden_size = num_channels // 8 
        head_size = hidden_size // self.nH   
        if self.masks == "causal":
            masks = self.trianglular_mask(location_num, downsampled_num, self.shrink, batch_size=batch_size)
            
        elif self.masks == "sparse":
            masks = self.get_grid_masks(h, h // 2)[:, :, self.shrink:]
           
        theta = self.conv1x1_theta(x[:, :, self.shrink:])
        theta = theta.view(-1, self.nH, head_size, location_num - self.shrink)

        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi) 
        phi = phi.view(-1, self.nH, head_size, downsampled_num)

        attn = torch.einsum('abcd, abce -> abde', theta, phi)
        
        if self.masks:
            adder = (1.0 - masks) * (-1000.0)
            adder = torch.from_numpy(adder).to(device=attn.device)
            attn += adder

        attn = F.softmax(attn, dim=-1)

        g = self.conv1x1_g(x)
        g = self.maxpool(g)

        g_hidden = num_channels // 2
        g_head_size = g_hidden // self.nH

        g = g.view(-1, self.nH, g_head_size, downsampled_num)
        attn_g = torch.einsum('abcd, abed -> abec', attn, g)

        attn_g = attn_g.reshape(-1, num_channels//2, h-self.shrink)
        attn_g = self.conv1x1_attn(attn_g)
        x = x[:, :, self.shrink:]
        out = x + self.sigma * attn_g

        return out
    
    def get_grid_masks(self, gridO, gridI):

        masks = []
        masks.append(sparse.get_grid_masks(gridO, gridI, nH=8))

        return np.array(masks)

    def trianglular_mask(self, long, short, shrink, batch_size):
        mask = np.zeros([long, short], dtype=bool)
        for i in range(long):
            mask[i, max(0, int(i/long*short)-30):int(i/long*short)+1] = True
        return repeat(mask, 'l s -> b n l s', b=batch_size, n=1)[:,:,shrink:,:]