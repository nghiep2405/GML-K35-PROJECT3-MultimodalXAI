"""Checkpoint-compatible edge-aware GAT."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EdgeAwareGAT(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, heads=4, dropout=0.2,
                 edge_mode='real_edges'):
        super().__init__()
        if edge_mode not in {'real_edges', 'no_edge', 'random_edges'}:
            raise ValueError(f'Unsupported edge mode: {edge_mode}')
        assert hidden_dim % heads == 0
        self.dropout = dropout
        self.edge_mode = edge_mode
        self.node_projection = nn.Linear(input_dim, hidden_dim)
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.edge_projection = nn.Linear(input_dim + 1, hidden_dim)
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.gat1 = GATConv(
            hidden_dim, hidden_dim // heads, heads=heads,
            edge_dim=hidden_dim, dropout=dropout, add_self_loops=False
        )
        self.gat2 = GATConv(
            hidden_dim, hidden_dim // heads, heads=heads,
            edge_dim=hidden_dim, dropout=dropout, add_self_loops=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_graph(self, batch):
        h = self.node_projection(batch['x']) + self.type_embedding(batch['node_type'])
        edge_attr = self.edge_projection(
            torch.cat([batch['edge_attr'], batch['edge_reverse']], dim=1)
        )
        edge_index = batch['edge_index']
        if self.edge_mode == 'no_edge':
            # Basic edge-removal ablation: preserve the architecture, node inputs,
            # optimizer, and residual path while removing every graph message.
            edge_index = edge_index[:, :0]
            edge_attr = edge_attr[:0]
        update = self.gat1(h, edge_index, edge_attr=edge_attr)
        h = self.norm1(h + F.dropout(F.elu(update), self.dropout, self.training))
        update = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h + F.dropout(F.elu(update), self.dropout, self.training))
        return F.normalize(h[batch['root_local']], dim=1)

    def encode_query(self, query_features):
        return F.normalize(self.query_projection(query_features), dim=1)
