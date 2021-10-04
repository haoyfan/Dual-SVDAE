import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from networks.GCN import GCN,MLP
import torch.nn.functional as F
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def forward(self, z, activation=None):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())

        return activation(adj)

# class Dual_AE(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  in_sfeats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(Dual_AE, self).__init__()
#         #self.g = g
#
#         self.Encoder = MLP(in_feats,
#                              n_hidden,
#                              n_classes,
#                              n_layers,
#                              activation,
#                              dropout)
#         self.A_decoder = MLP(n_classes,
#                              n_hidden,
#                              in_feats,
#                              n_layers,
#                              activation,
#                              dropout)
#         self.S_decoder = MLP(n_classes,
#                              n_hidden,
#                              in_sfeats,
#                              n_layers,
#                              activation,
#                              dropout)
#         self.InnerProducter = InnerProductDecoder()
#     def forward(self, g, features):
#         # n = features.shape[0]
#         # m = features.shape[1]
#         # h = (torch.eye(n,m).cuda())
#         adj = g.adjacency_matrix().to_dense().cuda()
#         A_z=self.Encoder(features)
#         S_z = torch.matmul(adj,A_z)
#        #S_z = self.S_encoder(g, features)
#         A_recon=self.A_decoder(A_z)
#         S_recon = self.S_decoder(S_z)
#         #S_recon = torch.matmul(S_z, S_z.t())
#         return A_z,S_z, A_recon, S_recon

class Dual_AE(nn.Module):
    def __init__(self,
                 in_feats,
                 in_sfeats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(Dual_AE, self).__init__()
        #self.g = g

        self.encoder = nn.ModuleList()
        # input layer
        self.encoder.append(nn.Sequential(nn.Conv1d(1, 16, kernel_size = 4, stride=2,padding = 1),nn.ReLU(inplace=True)))
        # hidden layers
        self.encoder.append(nn.Sequential(nn.Conv1d(16, 32,kernel_size= 4,stride= 2,padding = 1),nn.ReLU(inplace=True)))
        self.encoder.append(nn.Sequential(nn.Conv1d(32, 64,kernel_size= 4,stride= 2,padding = 1),nn.ReLU(inplace=True),
                                          nn.AdaptiveAvgPool1d(1)))
        # output layer

        self.Last_layers = nn.Linear(64,n_classes)

        self.A_decoder = MLP(n_classes,
                             n_hidden,
                             in_feats,
                             n_layers,
                             activation,
                             dropout)
        self.S_decoder = MLP(n_classes,
                             n_hidden,
                             in_sfeats,
                             n_layers,
                             activation,
                             dropout)

        self.dropout = nn.Dropout(p=dropout)
    def forward(self, g, features):

        adj = g.adjacency_matrix().to_dense()
        adj = adj.unsqueeze(dim = 1).cuda()
        features = features.unsqueeze(dim = 1)
        for i, layer in enumerate(self.encoder):
            features = layer(features)
        A_z = features.squeeze()
        A_z = self.Last_layers(A_z)
        for i, layer in enumerate(self.encoder):
            adj = layer(adj)
        S_z = adj.squeeze()
        S_z = self.Last_layers(S_z)
        A_recon=self.A_decoder(A_z)
        S_recon = self.S_decoder(S_z)
        #S_recon = torch.matmul(S_z, S_z.t())
        return A_z,S_z, A_recon, S_recon







