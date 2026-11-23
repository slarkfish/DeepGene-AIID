import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class WeightedGraphSAGEConv(MessagePassing):
    """Weighted GraphSAGE layer for undirected STRING PPI network."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)

        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, col, edge_weight)
        norm = edge_weight / (deg[col].clamp(min=1e-8))

        out_neigh = self.propagate(edge_index, x=x, norm=norm)
        out = self.lin_self(x) + self.lin_neigh(out_neigh)
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * x_j


class WeightedGraphSAGE(nn.Module):
    """Two-layer Weighted GraphSAGE encoder (Channel A: Undirected PPI)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = WeightedGraphSAGEConv(in_channels, hidden_channels)
        self.conv2 = WeightedGraphSAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class DiGCNLayer(MessagePassing):
    """Directed Graph Convolutional Network (DiGCN) layer for OmniPath directed graph."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_in = nn.Linear(in_channels, out_channels)
        self.lin_out = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def _normalize_adj(self, edge_index: torch.Tensor, num_nodes: int, edge_weight: torch.Tensor):
        row, col = edge_index
        deg_in = degree(col, num_nodes, dtype=edge_weight.dtype).clamp(min=1.0)
        deg_out = degree(row, num_nodes, dtype=edge_weight.dtype).clamp(min=1.0)
        norm = edge_weight / torch.sqrt(deg_out[row] * deg_in[col])
        return norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        num_nodes = x.size(0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)

        # Forward (incoming) direction
        norm_fwd = self._normalize_adj(edge_index, num_nodes, edge_weight)
        h_in = self.propagate(edge_index, x=self.lin_in(x), norm=norm_fwd)

        # Reverse (outgoing) direction
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        norm_rev = self._normalize_adj(rev_edge_index, num_nodes, edge_weight)
        h_out = self.propagate(rev_edge_index, x=self.lin_out(x), norm=norm_rev)

        return self.lin_self(x) + 0.5 * (h_in + h_out)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * x_j


class DiGCNEncoder(nn.Module):
    """Two-layer DiGCN encoder (Channel B: Directed OmniPath Graph)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = DiGCNLayer(in_channels, hidden_channels)
        self.conv2 = DiGCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class DualChannelEncoder(nn.Module):
    """
    Dual-channel graph encoder combining:
      - Channel A: Weighted GraphSAGE on undirected STRING PPI network
      - Channel B: DiGCN on directed OmniPath signaling network
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.channel_a = WeightedGraphSAGE(in_channels, hidden_channels, out_channels, dropout)
        self.channel_b = DiGCNEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.fusion_proj = nn.Linear(out_channels * 2, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        ppi_edge_index: torch.Tensor,
        dir_edge_index: torch.Tensor,
        ppi_edge_weight: torch.Tensor = None,
        dir_edge_weight: torch.Tensor = None,
    ):
        h_undir = self.channel_a(x, ppi_edge_index, ppi_edge_weight)
        h_dir = self.channel_b(x, dir_edge_index, dir_edge_weight)
        h_cat = torch.cat([h_undir, h_dir], dim=-1)
        return F.relu(self.fusion_proj(h_cat))
