"""classes CausalConv1d, EncoderBlock and DecoderBlock are not used
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.local_attention import LocalAttn
from src.utils.vqgan_utils import positional_encoding

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                                           padding=0, dilation=dilation, groups=groups, bias=bias) 
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))   

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding="same", dilation=1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels), nn.LeakyReLU(0.2), 
            nn.utils.spectral_norm(nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation)),
            nn.BatchNorm1d(in_channels), nn.LeakyReLU(0.2), 
            nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        )
        self.conv0 = nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.conv0(x) + self.conv(x)

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding="same", dilation=1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_channels), nn.LeakyReLU(0.2), 
            nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)),
            nn.BatchNorm1d(in_channels), nn.LeakyReLU(0.2), 
            nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation))
        )
        self.conv0 = nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.conv0(x) + self.conv(x)
    
class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(embed_dim, embed_dim*4, 1, 1, padding='same')),
            nn.GELU(),  # nice
            nn.utils.spectral_norm(nn.Conv1d(embed_dim*4, embed_dim, 1, 1, padding='same')),
        )
    def forward(self, x):
        return x + self.mlp(x)

class TransformerBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, embed_dim, shrink=0, masks=None):
        super().__init__()
        self.transformer = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            LocalAttn(embed_dim, shrink, masks),
            nn.BatchNorm1d(embed_dim),
            MLP(embed_dim=embed_dim)
        )
    def forward(self, x):
        return self.transformer(x)
    
class TransDiscriminator(nn.Module):
    def __init__(self, cfg, n_layers=4, skip=False, pos_enc=False):
        super(TransDiscriminator, self).__init__()
        self.skip = skip
        self.pos_enc = pos_enc
        self.seq_len = cfg["seq_len"]
        self.startblock = nn.Conv1d(cfg["dim"], cfg["latent_dim"], 3, 1, padding="same")
        if self.pos_enc: self.positional_encoding = positional_encoding(cfg["seq_len"], cfg["latent_dim"])
        self.outblock = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(cfg["latent_dim"], cfg["latent_dim"], 1),
            nn.Flatten(), nn.utils.spectral_norm(nn.Linear(cfg["latent_dim"] * cfg["seq_len"], 1)))
        self.conv_blocks = nn.ModuleList([
            TransformerBlock(cfg["latent_dim"], shrink=0, masks=0) for _ in range(n_layers)])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(cfg["latent_dim"], cfg["latent_dim"], kernel_size=1, stride=1, padding="same") for _ in range(n_layers)])

    def forward(self, x):
        skip_connections = []
        x = self.startblock(x)
        if self.pos_enc: x += self.positional_encoding[:, :, :self.seq_len]
        for i in range (len(self.conv_blocks)):
            skip_connections.append(self.skip_convs[i](x))
            x = self.conv_blocks[i](x)
        skip_connections.append(x)
        if self.skip:
            x = sum(skip_connections)
        y = self.outblock(x)
        return y