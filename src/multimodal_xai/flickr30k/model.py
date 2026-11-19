"""Checkpoint-compatible Flickr30k model."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

from .graph import EDGE_DIM


class SpatialGAT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gat1 = GATConv(512, 256, heads=2, concat=False, edge_dim=EDGE_DIM, add_self_loops=False)
        self.gat2 = GATConv(256, 512, heads=1, concat=False, edge_dim=EDGE_DIM, add_self_loops=False)
        self.graph_norm = nn.LayerNorm(512)
        self.query_projection = nn.Linear(512, 512)
        nn.init.zeros_(self.query_projection.bias)

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        hidden = F.elu(self.gat1(batch.x, batch.edge_index, batch.edge_attr))
        encoded = self.gat2(hidden, batch.edge_index, batch.edge_attr)
        roots = batch.ptr[:-1]
        return F.normalize(self.graph_norm(batch.x[roots] + encoded[roots]), dim=-1)

    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(query + self.query_projection(query), dim=-1)
