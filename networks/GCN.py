"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
#from dgl.nn.pytorch.conv import SAGEConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,  bias=False, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_hidden2,
                 n_layers,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            nn.Sequential(nn.Linear(in_feats, n_hidden),
                          nn.ReLU(inplace=True)))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(nn.Linear(n_hidden, n_hidden),
                              nn.ReLU(inplace=True)))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_hidden2))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
        return h
