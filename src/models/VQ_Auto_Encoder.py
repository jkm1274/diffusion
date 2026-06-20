import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from einops import repeat
from tqdm import tqdm
from vector_quantize_pytorch import FSQ, VectorQuantize
import torch.utils.data as data

from src.layers.vqgan_helpers import TransformerBlock, TransDiscriminator
from src.utils.vqtrans_utils import (
    calc_gradient_penalty, kl_anneal, load_data, 
    positional_encoding, recursive_simulator) 
from src.utils.loss import Loss 

# Configuration for training
dim = 12
# cfg = {
#   "dim": dim,
#   "z_dim": 4,
#   "latent_dim": 128,
#   "lr_rate": 6.e-6,
#   "batch_size": 128,
#   "num_epochs": 200,
#   "seq_len": 128,
#   "gpt_batch_size": 128,
#   "gpt_seq_len": 128,
#   "gpt_num_epochs": 400,
#   "gpt_lr": 4.5e-04,
#   "vqmethod": "fsq",
# }
cfg = {
    "vqmethod": "fsq",
    "dim": 12,
    "z_dim": 4,
    "latent_dim": 64,
    "lr_rate": 1.2315571723666024e-05,
    "batch_size": 128,
    "num_epochs": 50,
    "seq_len": 128,
    "gpt_seq_len": 128,
    "gpt_batch_size": 128,
    "gpt_lr": 4.473636174621264e-05,
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "gpt_num_epochs": 100,
    "vocab_size": 1000
}

# evaluation configs
num_examples_to_generate = 512
steps = 26 # 26

class TransEncoder(nn.Module):
    def __init__(self, latent_dim, n_layers=4, pos_enc=False, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = pos_enc
        if self.pos_enc: self.positional_encoding = positional_encoding(cfg["seq_len"], latent_dim)
        self.startblock = nn.Conv1d(cfg["dim"], latent_dim, 5, 1, padding="same")
        self.outblock = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(latent_dim, latent_dim, 3, padding="same"),
            nn.LeakyReLU(0.2), nn.Conv1d(latent_dim, cfg["z_dim"], 3, padding="same"))
        self.conv_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, shrink=0, masks=None) for i in range(n_layers)])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1, stride=1, padding="same") for _ in range(n_layers)])

    def forward(self, x):
        skip_connections = []
        x = self.startblock(x)
        if self.pos_enc: x += self.positional_encoding[:, :, :cfg["seq_len"]].to(device=x.device)
        for i in range (len(self.conv_blocks)):
            skip_connections.append(self.skip_convs[i](x))
            x = self.conv_blocks[i](x)
        skip_connections.append(x)
        x = sum(skip_connections)
        y = self.outblock(x)
        return y

class TransDecoder(nn.Module):
    def __init__(self, latent_dim, n_layers=4, skip=False, cfg=None):
        super(TransDecoder, self).__init__()
        self.cfg = cfg
        self.skip = skip
        self.startblock = nn.Sequential(
            nn.Conv1d(cfg["z_dim"], latent_dim, 3, 1, padding="same"),)
        self.outblock = nn.Sequential(
            nn.LeakyReLU(0.2), nn.Conv1d(latent_dim, latent_dim, 3, padding="same"),
            nn.LeakyReLU(0.2), nn.Conv1d(latent_dim, cfg["dim"], 3, padding="same"))
        self.conv_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, shrink=0, masks="causal") for i in range(n_layers)])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1, stride=1, padding="same") for _ in range(n_layers)])

    def forward(self, x):
        skip_connections = []
        x = self.startblock(x)
        for i in range (len(self.conv_blocks)):
            skip_connections.append(self.skip_convs[i](x))
            x = self.conv_blocks[i](x)
        skip_connections.append(x)
        if self.skip: x = sum(skip_connections)
        y = self.outblock(x)
        return y

class VQAutoencoder(nn.Module):
    def __init__(self, vq_method="fsq"):
        super(VQAutoencoder, self).__init__()
        self.cfg = cfg

        if vq_method in ["vq", "rvq"]:
            self.inplace_codebook_optim = torch.optim.Adam
            
        if vq_method == "vq":
            self.codebook = VectorQuantize(
                dim=cfg["z_dim"], codebook_size=cfg["num_codebook_vectors"],
                decay=0.8, learnable_codebook=False, ema_update=True, use_cosine_sim=True)
        elif vq_method == "fsq":
            self.codebook = FSQ([8, 5, 5, 5])     

        self.encoder = TransEncoder(latent_dim=cfg["latent_dim"], cfg=cfg)
        self.decoder = TransDecoder(latent_dim=cfg["latent_dim"], cfg=cfg)

    def forward(self, x):
        x = self.encoder(x) # b,c,l
        x = x.permute(0,2,1) # b, c, l -> b, l, c
        x, codebook_indices = self.codebook(x)[:2] 
        x = x.permute(0,2,1)
        x = self.decoder(x)
        return x, codebook_indices

    def encode(self, x):
        x = self.encoder(x) # b,c,l
        x = x.permute(0,2,1) # b, c, l -> b, l, c
        z, codebook_indices = self.codebook(x)[:2]         
        return z, codebook_indices

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat
    
    def save_model(self, model_path, epoch=None, optimizer=None):
        """
        Saves the model state, arguments, and optionally optimizer state and epoch.
        
        Args:
            model_path (str): Directory to save the model.
            epoch (int, optional): Current epoch to save for resuming training. Default is None.
            optimizer (torch.optim.Optimizer, optional): Optimizer to save its state. Default is None.
        """
        print('-' * 50)
        print(f'Saving model to {model_path}...')

        # Ensure the directory exists
        os.makedirs(model_path, exist_ok=True)

        # Generate a checkpoint filename that includes the epoch number
        checkpoint_filename = f"vqtrans_ae_checkpoint_epoch_{epoch}.pth" if epoch is not None else "checkpoint.pth"
        checkpoint_path = os.path.join(model_path, checkpoint_filename)

        # Save the model's state dictionary
        checkpoint = {"model_state_dict": self.state_dict()}  # Use self.state_dict() here

        # Add optional optimizer state and epoch to the checkpoint
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        print(f'Model successfully saved as {checkpoint_filename}!')
        print('-' * 50)

def trainAE(model, optimizer, cfg, discriminator=False):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 70, 130, 180], gamma=0.8, verbose=True)
    
    for epoch in range(cfg["num_epochs"]):
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
                # Convert x to float32
                x = x.to(device=device).float()

                # Forward pass
                x_reconst = model(x)[0]
                reconst_loss = torch.sum(torch.pow(x - x_reconst, 2))

                # Backprop and optimize
                loss = reconst_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix({
                    "epoch": epoch, "loss": loss.item(), 
                    "rec": float(reconst_loss.detach().cpu())})

        lr_scheduler.step()
