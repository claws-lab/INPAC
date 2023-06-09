import argparse
import copy
from typing import Callable

import torch
from torch import nn as nn
from torch.nn import GRUCell
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import TimeEncoder
from torch_scatter import scatter_max

import const


class Memory(TGNMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable, args: argparse.Namespace,
                 **kwargs):

        self.args = args
        self.mappings = kwargs.get('mappings')
        self.num_resource = len(self.mappings['resource2idx'])
        self.num_subreddit = len(self.mappings['subreddit2idx'])

        super(TGNMemory, self).__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        last_update = torch.empty(self.num_nodes, dtype=torch.long)

        self.lin1 = nn.Linear(args.resource_embedding_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.lin2 = nn.Linear(128, args.embedding_dim)

        self.lin1_channel = nn.Linear(args.resource_embedding_dim,
                                      args.embedding_dim)
        self.lin2_channel = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.lin3 = nn.Linear(args.embedding_dim * 2, args.embedding_dim)

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        self.register_buffer('last_update', last_update)
        self.register_buffer('__assoc__',
                             torch.empty(num_nodes, dtype=torch.long))
        self.last_update = self.last_update.to(self.args.device)
        self.__assoc__ = self.__assoc__.to(self.args.device)

        self.memory = self.memory.to(self.args.device2)

        self.msg_s_store = {}
        self.msg_d_store = {}

        # video nodes that we have already computed the embeddings for
        self.nodes_already_updated = set()

        self.reset_parameters()

        # Load the pretrained resource / video Embeddings
        if isinstance(args.path_resource_embeds, str):
            print("\tLoading pretrained subreddit / video embeddings")

            # Last resource embedding is for UNK

            (resource_embeds, channel_embeds, resource2channel) = torch.load(
                args.resource_metadata)

            self.resource2channel: dict = resource2channel

        else:
            resource_embeds = torch.randn(self.num_resource + 1,
                                          args.resource_embedding_dim).to(args.device)
            channel_embeds = torch.randn(self.num_resource + 1,
                                         args.resource_embedding_dim).to(args.device)

            self.resource2channel = {i: i for i in range(self.num_resource + 1)}

        self.resource_embeddings = nn.Embedding.from_pretrained(resource_embeds,
                                                                sparse=True)
        self.channel_embeddings = nn.Embedding.from_pretrained(channel_embeds,
                                                               sparse=True)
        if args.verbose:
            print("\tInitializing subreddit / video embeddings")
            print("\t[Embed] Size of Resource Embeds",
                  self.resource_embeddings.weight.element_size() * self.resource_embeddings.weight.nelement())
            print("\t[Embed] Size of Channel Embeds",
                  self.channel_embeddings.weight.element_size() * self.channel_embeddings.weight.nelement())

        # Put the resource / channel embeddings in the GPU memory

    def __del__(self):
        del self.mappings, self.args

    def reset_state(self):
        super().reset_state()
        self.nodes_already_updated = set()

    def __update_memory__(self, n_id):
        memory, last_update = self.__get_updated_memory__(n_id)

        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def __get_updated_memory__(self, n_id):

        """
        Inherited from TGNMemory.__get_updated_memory__. The difference is that self.memory can be on another device
        :param n_id:
        :return:
        """

        n_id = n_id.to(self.args.device)

        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self.__compute_msg__(
            n_id, self.msg_s_store, self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self.__compute_msg__(
            n_id, self.msg_d_store, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)

        BATCH_SIZE = self.args.batch_size * 4

        if n_id.size(0) <= BATCH_SIZE:
            # else:

            # When we perform memory.eval() during test, aggr should be all zeros
            aggr = self.aggr_module(msg, self.__assoc__[idx], t, n_id.size(0),
                                    args=self.args)
            memory = self.gru(aggr, self.memory[n_id.to(self.args.device2)].to(
                self.args.device))

        else:

            # If we can fit into memory, we can directly use the GPU
            aggr = self.aggr_module(msg, self.__assoc__[idx], t, n_id.size(0),
                                    args=self.args)
            memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter_max(t, idx, dim=0, dim_size=dim_size)[0][n_id]

        return memory, last_update

    def __compute_msg__(self, n_id, msg_store, msg_module):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))

        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        return msg, t, src, dst

    def set_subreddit_embeddings(self, model_session: nn.Module):
        indices = torch.arange(1, self.num_subreddit + 1).to(self.args.device)
        subreddit_embeddings = model_session.embedding(indices)

        self.memory[self.num_resource:] = subreddit_embeddings
        print(f"\tsubreddit embeds: {self.memory.mean()}")

    def compute_resource_embeds(self, node_ids=torch.Tensor):
        BATCH_SIZE = 8192

        channel_ids = torch.tensor(
            [self.resource2channel[x] for x in node_ids.cpu().numpy()],
            dtype=torch.long, device=self.args.device)

        for idx_resource in range(0, len(node_ids), BATCH_SIZE):

            # (#resources, #embedding_dim)
            resource_embeds_batch = self.resource_embeddings(
                node_ids[idx_resource: idx_resource + BATCH_SIZE] + 1)

            channel_embeds_batch = self.channel_embeddings(
                channel_ids[idx_resource: idx_resource + BATCH_SIZE] + 1)

            resource_embeds = self.lin1(resource_embeds_batch)
            resource_embeds = self.dropout(self.relu(resource_embeds))
            resource_embeds = self.lin2(resource_embeds)

            # (#resources, #embedding_dim)
            channel_embeds = self.lin1_channel(channel_embeds_batch)
            channel_embeds = self.dropout(self.relu(channel_embeds))
            channel_embeds = self.lin2_channel(channel_embeds)

            if self.args.video_channel_embed_aggregation_method == const.CONCAT:
                # (#resources, #embedding_dim * 2)
                resource_embeds = torch.cat(
                    [resource_embeds, channel_embeds], dim=1)

                # (#resources, #embedding_dim)
                resource_embeds = self.lin3(resource_embeds)

            elif self.args.video_channel_embed_aggregation_method == const.ADD:
                # (#resources, #embedding_dim)
                resource_embeds = resource_embeds + channel_embeds

            elif self.args.video_channel_embed_aggregation_method == const.MUL:
                # (#resources, #embedding_dim)
                resource_embeds = resource_embeds * channel_embeds

            else:
                raise ValueError(
                    f"Unknown aggregation method for video & channel embeddings: {self.args.video_channel_embed_aggregation_method}")

            return resource_embeds
