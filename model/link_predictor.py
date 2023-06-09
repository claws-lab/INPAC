import torch
from torch import nn
from torch.nn import Linear

import const


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, args):
        super().__init__()
        self.args = args
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.aggregation = args.link_prediction_aggregation_method
        if self.aggregation in [const.ADD, const.MUL]:
            self.lin_final = Linear(in_channels, 1)
        elif self.aggregation == const.CONCAT:
            self.lin_final = Linear(in_channels * 2, 1)
        elif self.aggregation == const.DOT:
            pass
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def __del__(self):
        if hasattr(self, 'args'):
            del self.args

    def forward(self, z_src, z_dst):
        if self.aggregation == const.DOT:

            # (#users, #items)
            h = z_src.matmul(z_dst.T)

            return h

        elif self.aggregation == const.ADD:
            """Default aggregation used in the paper."""

            h = self.lin_src(z_src) + self.lin_dst(z_dst)
            h = self.relu(h)

            h = self.lin_final(h)

            # Sigmoid is sometimes disabled as it will saturate at 0 and 1
            if self.args.loss == const.BCE:
                h = self.sigmoid(h)

            # h = self.sigmoid(h)

            return h
        elif self.aggregation == const.MUL:

            """
                        Added 2022.12.10
                        Aggregation used in the paper for the recommendation task.
                        """

            z_src = self.lin_src(z_src)
            z_dst = self.lin_dst(z_dst)
            z = torch.mul(z_src, z_dst)

            h = self.lin_final(self.dropout(self.relu(z)))

            # Sigmoid is sometimes disabled as it will saturate at 0 and 1

            if self.args.loss == const.BCE:
                h = self.sigmoid(h)

            return h

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
