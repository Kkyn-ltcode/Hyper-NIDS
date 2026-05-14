"""
KAIROS model components.

Adapted from:
  https://github.com/ProvenanceAnalytics/kairos/blob/main/DARPA/CADETS_E3/model.py

Components:
  - GraphAttentionEmbedding: 2-layer TransformerConv with time encoding
  - LinkPredictor: MLP that predicts edge type from (src, dst) embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import TransformerConv


class GraphAttentionEmbedding(nn.Module):
    """TGN-style graph attention with time-relative edge features."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels, heads=8,
            dropout=0.0, edge_dim=edge_dim
        )
        self.conv2 = TransformerConv(
            out_channels * 8, out_channels, heads=1, concat=False,
            dropout=0.0, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        device = x.device
        last_update = last_update.to(device)
        t = t.to(device)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(nn.Module):
    """MLP for edge type prediction from source and destination embeddings."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)
        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, in_channels // 2),
            nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels // 2, out_channels),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)
        return self.lin_seq(h)
