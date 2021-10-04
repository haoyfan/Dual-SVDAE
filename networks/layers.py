# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 15:28
# @Author  : Haoyi Fan
# @Email   : isfanhy@gmail.com
# @File    : layers.py
import torch
import math

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class GCN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=False, dropout=0.0, activation=F.relu):
        super(GCN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = nn.Dropout(p=dropout)
        self.activation=activation

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = self.dropout(x)
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return self.activation(x)


class Dense(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, activation=F.relu):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return self.activation(x)

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0.0, activation=F.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, z):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.dropout(z)
        adj = torch.matmul(z, z.t())
        return self.activation(adj)


class AttributeDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0.0, activation=lambda x: x):
        super(AttributeDecoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, z, z_T):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.dropout(z)
        z_T = self.dropout(z_T)

        adj = torch.matmul(z, z_T.t())

        return self.activation(adj)


