import math

import torch
from torch import nn
import torch.nn.functional as F

import const


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)

        # split whole x back into graphs G_i
        v_i = torch.split(session_embedding, tuple(
            sections.cpu().numpy()))

        # repeat |V|_i times for the last node embedding
        v_n_repeat = tuple(
            nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)

        # Eq(6)
        alpha = self.q(torch.sigmoid(
            self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(
                session_embedding)))  # |V|_i * 1
        s_g_whole = alpha * session_embedding  # |V|_i * hidden_size

        # split whole s_g into graphs G_i
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in
                    s_g_split) # tuple of length 1220

        # Eq(7). W_3: 128 -> 64
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i) # tuple of length 1220
        s_h = self.W_3(
            torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1)) # (1220, 64)

        # Eq(8)
        z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))

        return z_i_hat


class GNNModel(nn.Module):
    """
    Static Modeling in Section 3.2 of KDD 2023 Paper

    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """

    def __init__(self, hidden_size, n_node, args):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node + 1, self.hidden_size)

        # You can try other GNN layers here

        if args.session_graph_operator == const.GATEDGRAPHCONV:
            from torch_geometric.nn import GatedGraphConv
            self.gnn = GatedGraphConv(self.hidden_size, num_layers=1)

        elif args.session_graph_operator == const.APPNP:
            from torch_geometric.nn import APPNP

            # This is the default setting in the [original APPNP paper](https://arxiv.org/pdf/1810.05997.pdf)
            self.gnn = APPNP(K=10, alpha=0.1)

        else:
            raise NotImplementedError

        self.e2s = Embedding2Score(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def get_embeddings(self, indices):
        embeddings = self.embedding(indices)
        return embeddings


    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x - 1, data.edge_index, data.batch, data.edge_weight

        embedding = self.embedding(x).squeeze()
        hidden = self.gnn(embedding, edge_index, edge_weight=edge_weight)
        hidden2 = F.relu(hidden)

        return self.e2s(hidden2, self.embedding, batch), hidden
