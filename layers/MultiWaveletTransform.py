import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math
from functools import partial
from torch import nn, einsum, diagonal
from math import log2, ceil
import pdb
from layers.utils import get_filter


class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """
    def __init__(self, ich=512, k=8, alpha=16, c=64, nCZ=1, L=0, base='legendre'):
        super(MultiWaveletTransform, self).__init__()
        print('MutliWavelet enhanced block used!')
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ2d(k, alpha, c, L, base) for i in range(nCZ))

    def forward(self, x):
        B, L, H, E = x.shape

        x = x.view(B, L, -1)
        x = self.Lk0(x).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            if i < self.nCZ - 1:
                x = F.relu(x)

        x = self.Lk1(x.view(B, L, -1))
        x = x.view(B, L, -1, E)
        return x.contiguous()


class MWT_CZ2d(nn.Module):
    def __init__(self, k, alpha, c, L, base='legendre', **kwargs):
        super(MWT_CZ2d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT2d(k, alpha, c)
        self.B = sparseKernelFT2d(k, alpha, c)
        self.C = sparseKernelFT2d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(  # 将后面的tensor注册到模型的buffer属性中，方便再次使用
            np.concatenate((H0.T, H1.T), axis=0)))  # concatenate沿着轴连接序列
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

        # # 用可学习矩阵代替原来计算得来的矩阵
        # self.scale = (1 / (2 * k ** 2))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(2 * k, k, dtype=torch.float32))  # rand:返回指定区间上的均匀分布，生成三维tensor，维度为[c*k,c*k,modes1]
        # self.weights1.requires_grad = True
        # self.weights2 = nn.Parameter(self.scale * torch.rand(2 * k, k, dtype=torch.float32))
        # self.weights2.requires_grad = True
        # self.weights3 = nn.Parameter(self.scale * torch.rand(2 * k, k, dtype=torch.float32))
        # self.weights3.requires_grad = True
        # self.weights4 = nn.Parameter(self.scale * torch.rand(2 * k, k, dtype=torch.float32))
        # self.weights4.requires_grad = True

    def forward(self, x):
        B, L, c, k = x.shape
        ns = math.floor(np.log2(L))  # 即整个分解/重建结构迭代次数
        nl = pow(2, math.ceil(np.log2(L)))  # pow幂函数
        extra_x = x[:, 0:nl - L, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = torch.jit.annotate(List[Tensor], [])   # 定义了一个tensor
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):    # start,stop,step
            x = x + Us[i]   # 重构中两tensor相加部分
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :L, :, :]
        return x

    def wavelet_transform(self, x):  # 分解滤波转换
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)     # 公式(6)
        s = torch.matmul(xa, self.ec_s)     # 公式(7)
        # d = torch.matmul(xa, self.weights1)  # 公式(6)
        # s = torch.matmul(xa, self.weights2)  # 公式(7)
        return d, s

    def evenOdd(self, x):   # 重构滤波转化
        B, N, c, ich = x.shape  # (B, T, N, c, k)
        assert ich == 2 * self.k    # 重构前k相当于扩大了一倍(*2)
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        # x_e = torch.matmul(x, self.weights3)
        # x_o = torch.matmul(x, self.weights4)

        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class sparseKernelFT2d(nn.Module):
    def __init__(self, k, alpha, c, nl=1, **kwargs):
        super(sparseKernelFT2d, self).__init__()
        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c*k, c*k, self.modes1, dtype=torch.cfloat))    # rand:返回指定区间上的均匀分布，生成三维tensor，维度为[c*k,c*k,modes1]
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, L, c, k = x.shape  # (B, L, c, k)

        x = x.view(B, L, -1).permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, L // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c*k, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft)
        x = x.permute(0, 2, 1).view(B, L, c, k)
        return x


class TransformLayer(nn.Module):
    """
    线性层过滤q、k、v，再调用各个块(傅里叶、小波，共四个)处理q、k、v，最终线性层调整输出
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(TransformLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)     # 若d_keys没有值传进来，就用后一个值；d_model=16,n_heads=8
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation    # 傅里叶增强快...
        self.in_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads

        value = self.in_projection(x).view(B, L, H, -1)

        out = self.inner_correlation(value)

        out = out.view(B, L, -1)
        return self.out_projection(out)


class Feedforward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.05, activation='relu'):
        # Implementation of Feedforward model
        super().__init__()

        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward, kernel_size=1, bias=False)
        # self.dropout = nn.Dropout(dropout)
        # self.conv2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model, kernel_size=1, bias=False)
        # self.activation = F.relu() if activation == "relu" else F.gelu
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu() if activation == "relu" else F.gelu

    def forward(self, x):
        # y = x
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # x = x + y
        # return x
        y = x
        y = self.dropout1(self.activation(self.linear1(x)))
        y = self.dropout2(self.linear2(y))
        x = x + y
        return x
