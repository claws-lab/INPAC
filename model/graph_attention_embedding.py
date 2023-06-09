import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GATConv


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, args, in_channels, out_channels,
                 time_enc, msg_dim=None):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        if args.do_static_modeling:
            edge_dim += 2 * args.embedding_dim
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

        # You can try different Graph Convolutional Operators here, such as:
        # self.conv = GATConv(in_channels, out_channels // 2, heads=2,
        #                             dropout=0.1, edge_dim=edge_dim)


    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype)) # #Timestamps -> (#Timestamps, 64)
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


