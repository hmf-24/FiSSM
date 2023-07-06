import sys
import os

import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from utils import unroll
from utils.op import transition
import pickle
import pdb
from einops import rearrange, repeat
import opt_einsum as oe
from layers.MultiWaveletTransform import MultiWaveletTransform, TransformLayer, Feedforward
from layers.S4d import S4D

contract = oe.contract

if tuple(map(int, torch.__version__.split('.')[:2]))==(1, 11):
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2]))>=(1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TokenEmbedding(nn.Module):    # 本质就是一个一维卷积
    def __init__(self, c_in, d_model):  # c_in：默认7；d_model：默认512
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)    # circular：循环填充
        for m in self.modules():    # modules返回自定义的网络结构中的可迭代模块
            if isinstance(m, nn.Conv1d):    # 判断这个模块是不是卷积模块，若是就使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs, N=512, N2=32):
        super(Model, self).__init__()
        self.configs = configs
        self.c = configs.c
        self.k = configs.k
        self.prenorm = configs.prenorm
        self.n_layers = configs.S4D_layer
        # self.modes = configs.modes
        self.d_model = configs.d_model
        self.L = 3
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.seq_len_all = self.seq_len + self.label_len

        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.modes1 = min(configs.modes1, self.pred_len // 2)
        # self.modes1 = 32
        self.enc_in = configs.enc_in
        self.proj = False
        self.e_layers = configs.e_layers
        self.mode_type = configs.mode_type
        if self.configs.ours:
            # b, s, f means b, f
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        if configs.enc_in > 1000:
            self.proj = True
            self.conv1 = nn.Conv1d(configs.enc_in, configs.d_model, 1)
            self.conv2 = nn.Conv1d(configs.d_model, configs.dec_in, 1)
            self.d_model = configs.d_model
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.d_model))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.d_model))

        if self.configs.ab == 2:
            self.multiscale = [1, 2, 4]

            self.value_embedding = TokenEmbedding(c_in=configs.enc_in, d_model=configs.d_model)

            self.out_projection1 = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.out_projection2 = nn.Linear(self.pred_len, self.pred_len, bias=True)
            self.out_projection3 = nn.Linear(self.pred_len*2, self.pred_len, bias=True)
            self.out_projection4 = nn.Linear(self.pred_len * 4, self.pred_len, bias=True)
            self.out_projection5 = nn.Linear(self.seq_len, self.pred_len, bias=True)

            # 新增块2
            transformer = MultiWaveletTransform(ich=self.d_model, L=self.L, alpha=configs.modes1, base='legendre', c=self.c, k=self.k)
            self.multiwavelettransform = nn.ModuleList([TransformLayer(transformer, configs.d_model, configs.n_heads) for _ in range(len(self.multiscale))])

            # Stack S4 layers as residual blocks
            self.encoder = nn.Linear(configs.enc_in, self.d_model)
            self.s4_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(self.n_layers):
                self.s4_layers.append(
                    S4D(self.d_model, dropout=configs.dropout, transposed=True, lr=min(0.001, configs.learning_rate))
                )
                self.norms.append(nn.LayerNorm(self.d_model))
                self.dropouts.append(dropout_fn(configs.dropout))
            
            self.norm_wavelet = nn.LayerNorm(self.d_model)
            self.dropout = nn.Dropout(configs.dropout)
            self.feedforward = Feedforward(self.d_model, configs.d_ff, configs.dropout, configs.activation)

            self.mlp = nn.Linear(len(self.multiscale), 1)

    def forward(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        if self.configs.ab == 2:
            return_data = [x_enc]
            if self.proj:
                x_enc = self.conv1(x_enc.transpose(1, 2))
                x_enc = x_enc.transpose(1, 2)
            if self.configs.ours:
                means = x_enc.mean(1, keepdim=True).detach()
                # mean
                x_enc = x_enc - means
                # var
                stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                x_enc /= stdev
                # affine
                x_enc = x_enc * self.affine_weight + self.affine_bias
            B, L, E = x_enc.shape
            seq_len = self.seq_len
            x_decs = []
            jump_dist = 0
            for i in range(0, len(self.multiscale)):
                x_decs1 = []
                x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
                x_in = x_enc[:, -x_in_len:]

                # S4D
                x = self.encoder(x_in)  # (B, L, d_input) -> (B, L, d_model)

                x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
                for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
                    # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
                    z = x
                    if self.prenorm:
                        z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                    # Apply S4 block: we ignore the state input and output
                    z, _ = layer(z)
                    # Dropout on the output of the S4 block
                    z = dropout(z)
                    # Residual connection
                    x = z + x
                    if not self.prenorm:
                        x = norm(x.transpose(-1, -2)).transpose(-1, -2)

                # 新增1：多小波转换
                x = x.permute(0, 2, 1)
                out1 = self.multiwavelettransform[i](x)
                out1 = x + self.dropout(out1)
                out1 = self.feedforward(out1)
                out1 = self.norm_wavelet(out1)

                x_dec = self.out_projection1(out1).permute(0, 2, 1)

                m = x_dec.shape[2]
                if self.pred_len == m:
                    x_dec = self.out_projection2(x_dec)
                elif self.pred_len == m/2:
                    x_dec = self.out_projection3(x_dec)
                elif self.pred_len == m/4:
                    x_dec = self.out_projection4(x_dec)
                else:
                    x_dec = self.out_projection5(x_dec)

                x_decs += [x_dec]
            return_data.append(x_in)
            return_data.append(out1)
            x_dec = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0, 2, 1)
            if self.configs.ours:
                x_dec = x_dec - self.affine_bias
                x_dec = x_dec / (self.affine_weight + 1e-10)
                x_dec = x_dec * stdev
                x_dec = x_dec + means
            if self.proj:
                x_dec = self.conv2(x_dec.transpose(1, 2))
                x_dec = x_dec.transpose(1, 2)
            return_data.append(x_dec)
        if self.output_attention:
            return x_dec, return_data  
        else:
            return x_dec, None  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 2
        modes1 = 32
        seq_len = 60
        label_len = 0
        pred_len = 60
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 512
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        moving_avg = 25
        c_out = 7
        activation = 'gelu'
        wavelet = 0
        ours = False
        version = 0
        ratio = 1
        mode_type = 2
        attention_version = 'Wavelets'
        learning_rate = 0.0001
        prenorm = False
        S4D_layer = 3
        c = 64
        k = 8

    configs = Configs()
    model = Model(configs).to(device)

    enc = torch.randn([32, configs.seq_len, configs.enc_in]).cuda()
    enc_mark = torch.randn([32, configs.seq_len, 4]).cuda()

    dec = torch.randn([32, configs.label_len + configs.pred_len, configs.dec_in]).cuda()
    dec_mark = torch.randn([32, configs.label_len + configs.pred_len, 4]).cuda()
    out = model.forward(enc, enc_mark, dec, dec_mark)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print('model size', count_parameters(model))
    print('input shape', enc.shape)
    print('output shape', out[0].shape)  # out[0]返回x_dec
    a, b, c, d = out[1]     # out[1]返回return_data
    print('input shape', a.shape)
    print('hippo shape', b.shape)
    print('processed hippo shape', c.shape)
    print('output shape', d.shape)

