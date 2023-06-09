import os
import os.path as osp
import random
import sys
from collections import defaultdict
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import TemporalDataLoader, DataLoader
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastNeighborLoader,
)
from tqdm import tqdm

from dataset.cig_dataset import CIGDataset
from model.last_aggregator import LastAggregator
from model.sequential import GNNModel

sys.path.insert(0, osp.join(os.getcwd(), "src"))

from model.graph_attention_embedding import GraphAttentionEmbedding
from model.link_predictor import LinkPredictor
from model.memory import Memory
from utils.utils_logging import setup_logger

import const
from arguments import args, args_static_modeling
from dataset.reddit_dataset import RedditDataset

from utils.utils_eval import get_eval_df, save_results_to_excel, \
    get_ranking_results
from utils.utils_training import set_directories_from_args
from utils.utils_neg_sampling import sample_neg_items
from utils.utils_misc import project_setup

module_names = ['gnn', 'link_pred', 'memory', 'neighbor_loader',
                "model_static"]
optimizer_names = ['optimizer_dense', 'optimizer_sparse',
                   'optimizer_session']
lr_scheduler_names = ['scheduler_dense', 'scheduler_sparse',
                      'optimizer_session']

project_setup()
args.model = "INPAC"

def load_model(args, **kwargs):
    names = module_names + optimizer_names + lr_scheduler_names
    path = osp.join(args.model_full_dir,
                    f'{args.comment}_ep{args.load_checkpoint_from_epoch}.pt')
    print(f"\t[Load] Loading model from {path}")

    d = torch.load(path)

    for name in names:
        if kwargs.get(name) is not None:
            if isinstance(kwargs[name], (nn.Module, torch.optim.Optimizer,
                                         torch.optim.lr_scheduler._LRScheduler)):
                kwargs[name].load_state_dict(d[name])
            else:
                kwargs[name] = d[name]


def save_model(epoch, args, **kwargs):
    print(f"\t[Save] Epoch {epoch} saving model to {args.model_dir}")
    names = module_names + optimizer_names + lr_scheduler_names
    d = {}

    for name in names:
        if kwargs.get(name) is not None:
            if isinstance(kwargs[name], (nn.Module, torch.optim.Optimizer,
                                         torch.optim.lr_scheduler._LRScheduler)):
                d[name] = kwargs[name].state_dict()

            else:
                d[name] = kwargs[name]

    path = osp.join(args.model_full_dir, f'{args.comment}_ep{epoch}.pt')

    torch.save(d, path)


def train(neg_items_df: pd.DataFrame, epoch: int):
    memory.train()
    gnn.train()
    link_pred.train()

    if epoch == args.load_checkpoint_from_epoch:
        return

    # Start with a fresh memory.
    memory.reset_state()

    # Start with an empty graph.
    neighbor_loader.reset_state()

    if epoch == 0:
        return

    total_loss = 0.

    # Train for static modeling
    if args.do_static_modeling:

        model_static.train()

        mean_loss_session = 0.0
        updates_per_epoch = len(cig_train_loader)

        for i, batch in enumerate(tqdm(cig_train_loader,
                                       desc=f"Train Ep.{epoch} Static Modeling")):
            optimizer_session.zero_grad()
            scores, last_hidden_session = model_static(batch.to(args.device))
            targets = batch.y - 1
            loss_session = model_static.loss_function(scores, targets)

            loss_session.backward()
            optimizer_session.step()
            writer.add_scalar('Loss/SRGNN_step', loss_session.item(),
                              epoch * updates_per_epoch + i)

            if args.verbose and (i + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {i + 1} | Session Loss {loss_session.item()}")

            mean_loss_session += loss_session / batch.num_graphs

        writer.add_scalar('Loss/SRGNN_epoch', mean_loss_session, epoch)

    # Dynamic modeling
    for idx_batch, batch in enumerate(
            tqdm(train_loader, desc=f"Train Ep.{epoch} Dynamic Modeling")):
        batch = batch.to(args.device)
        optimizer_dense.zero_grad()
        if optimizer_sparse is not None:
            optimizer_sparse.zero_grad()

        # src is always smaller than dst
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes. The RANDOM method is the fastest.
        if args.train_sample_method == const.PER_INTERACTION:
            src_np = src.cpu().numpy()

            neg_dst = torch.tensor([random.sample(neg_items_df.loc[x].neg_items,
                                                  args.train_neg_sampling_ratio)
                                    for x
                                    in src_np], dtype=torch.long, device=device)

            # flatten list of list into list
            neg_dst = neg_dst.reshape(-1)

        elif args.train_sample_method == const.RANDOM:
            neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1,
                                    (src.size(0),),
                                    dtype=torch.long, device=device)
        else:
            raise NotImplementedError

        # unique nodes in all of source and destination nodes (pos and neg)
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        if args.do_static_modeling:
            """Use the concatenation of node representations as the message."""
            node_embeds = torch.empty((z.shape[0], args.embedding_dim),
                                      dtype=torch.float, device=args.device)

            node_mask = (n_id >= dataset.num_src)
            node_embeds[node_mask] = model_static.get_embeddings(
                n_id[node_mask] - dataset.num_src + 1)
            node_embeds[~node_mask] = memory.compute_resource_embeds(
                n_id[~node_mask])

            message_node1 = node_embeds[edge_index[0]]
            message_node2 = node_embeds[edge_index[1]]
            message = torch.cat([message_node1, message_node2], dim=1)

        else:
            """Use the original message"""
            message = data.msg[e_id].to(args.device)

        z = gnn(z.to(args.device), last_update, edge_index,
                data.t[e_id].to(args.device),
                message)

        z_pos_dst = z[assoc[pos_dst]]
        z_neg_dst = z[assoc[neg_dst]]

        pos_out = link_pred(z[assoc[src]], z_pos_dst)
        neg_out = link_pred(z[assoc[
            src.repeat(args.train_neg_sampling_ratio, 1).T.reshape(-1)]],
                            z_neg_dst)

        if args.loss == const.BCE:

            loss = torch.nn.BCELoss()(pos_out,
                                      torch.ones_like(pos_out))
            loss += torch.nn.BCELoss()(neg_out,
                                       torch.zeros_like(neg_out))

        elif args.loss == const.BPR:

            maxi = nn.LogSigmoid()(pos_out - neg_out)
            loss = -1 * torch.mean(maxi)

        else:
            raise NotImplementedError

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer_dense.step()
        if optimizer_sparse is not None:
            optimizer_sparse.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, epoch: int, neg_items_df: pd.DataFrame, eval_collection: dict,
         split: str = "",
         **kwargs):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    aps, aucs = [], []
    pred_and_true_df_li = []

    all_y_pred = []
    all_y_true = []

    full_li = []
    y_true_mat_li = []

    # ----------------- Static Modeling -----------------

    if args.do_static_modeling:
        model_static.eval()
        cig_eval_loader = kwargs['cig_eval_loader']

        mean_loss_session = 0.0

        for i, batch in enumerate(tqdm(cig_eval_loader,
                                       desc=f"Eval Ep.{epoch} Static Modeling")):
            scores, last_hidden_session = model_static(batch.to(args.device))
            targets = batch.y - 1
            loss_session = model_static.loss_function(scores, targets)

            if args.verbose and (i + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {i + 1} | Session Loss {loss_session.item()}")

            mean_loss_session += loss_session / batch.num_graphs

        writer.add_scalar('Loss/SRGNN_epoch', mean_loss_session, epoch)

    # ----------------- Dynamic Modeling -----------------

    desc = f"{split.capitalize()} Ep.{epoch}"
    for idx_batch, batch in tqdm(enumerate(loader), position=0, leave=True,
                                 total=len(loader),
                                 desc=desc):

        batch = batch.to(args.device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Note that eval set should be fixed
        src_np = src.cpu().numpy()

        assert (neg_items_df.neg_items.apply(
            len) == args.eval_neg_sampling_ratio).all()
        neg_dst = torch.tensor([neg_items_df.loc[x].neg_items for x
                                in src_np], dtype=torch.long, device=device)

        # flatten list of list into list
        neg_dst = neg_dst.reshape(-1)

        assert len(src) == len(pos_dst) == len(
            neg_dst) // args.eval_neg_sampling_ratio

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()

        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # z: (num_nodes, 64)
        z, last_update = memory(n_id)

        if args.do_static_modeling:
            """Use the concatenation of node representations as the message."""
            node_embeds = torch.empty((n_id.shape[0], args.embedding_dim),
                                      dtype=torch.float, device=args.device)

            node_mask = (n_id >= dataset.num_src)
            node_embeds[node_mask] = model_static.get_embeddings(
                n_id[node_mask] - dataset.num_src + 1)
            node_embeds[~node_mask] = memory.compute_resource_embeds(
                n_id[~node_mask])

            message_node1 = node_embeds[edge_index[0]]
            message_node2 = node_embeds[edge_index[1]]
            message = torch.cat([message_node1, message_node2], dim=1)

        else:
            """Use the original message"""
            message = data.msg[e_id].to(args.device)

        z = gnn(z.to(args.device), last_update, edge_index,
                data.t[e_id].to(args.device),
                message)

        # [Batch] Start: Eval using K Negative Samples

        if args.do_static_modeling:

            # This is setting the session graph embedding AFTER training on the dynamic graph
            pos_dst_session_graph = pos_dst + 1 - dataset.num_resource
            neg_dst_session_graph = neg_dst + 1 - dataset.num_resource
            assert min(pos_dst_session_graph) >= 1
            assert max(pos_dst_session_graph) <= dataset.num_subreddit

            z_pos_dst = z[assoc[pos_dst]] + model_static.embedding(
                pos_dst_session_graph)
            z_neg_dst = z[assoc[neg_dst]] + model_static.embedding(
                neg_dst_session_graph)

        else:

            z_pos_dst = z[assoc[pos_dst]]
            z_neg_dst = z[assoc[neg_dst]]

        pos_out = link_pred(z[assoc[src]], z_pos_dst)

        neg_out = link_pred(
            z[assoc[
                src.repeat(args.eval_neg_sampling_ratio, 1).T.reshape(-1)]],
            z_neg_dst)

        """
        1, 0, ..., 0 (K negatives), 1, 0, ..., 0, 1, 0, ..., 0
        """
        y_pred, y_true = [], []
        pos_out, neg_out = pos_out.squeeze().tolist(), neg_out.squeeze().tolist()
        dst_sorted = []

        pos_dst_li = pos_dst.tolist()
        neg_dst_li = neg_dst.tolist()

        for i in range(len(batch)):
            y_pred += [pos_out[i]] + neg_out[
                                     (i * args.eval_neg_sampling_ratio): (
                                             (
                                                     i + 1) * args.eval_neg_sampling_ratio)]
            y_true += [1] + [0] * args.eval_neg_sampling_ratio

            dst_sorted += [pos_dst_li[i]] + neg_dst_li[
                                            (
                                                    i * args.eval_neg_sampling_ratio): (
                                                    (
                                                            i + 1) * args.eval_neg_sampling_ratio)]

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Count correct predictions
        y_pred_label = (np.array(y_pred) > 0.5)

        all_y_pred += [y_pred]
        all_y_true += [y_true]

        # eval_index is for the convenience of calculating NDCG@K, Recall@K, etc.
        eval_index = np.arange(idx_batch * args.batch_size,
                               idx_batch * args.batch_size + len(batch))

        pred_and_true_df = pd.DataFrame(
            [src.cpu().numpy().repeat(args.eval_neg_sampling_ratio + 1),
             dst_sorted, y_pred,
             y_pred_label.squeeze().tolist(), y_true,
             eval_index.repeat(args.eval_neg_sampling_ratio + 1)]).T

        assert pred_and_true_df.isnull().sum().sum() == 0

        pred_and_true_df.columns = [const.SRC, const.DST, const.PRED,
                                    const.Y_PRED, const.Y_TRUE,
                                    const.EVAL_INDEX]

        pred_and_true_df[const.RESOURCE] = pred_and_true_df.src.apply(
            lambda x: dataset.mappings['idx2resource'][x])

        # Pad by #resources
        pred_and_true_df[const.SUBREDDIT] = pred_and_true_df.dst.apply(
            lambda x: dataset.mappings['idx2subreddit'][
                x - dataset.num_resource])

        pred_and_true_df = pred_and_true_df.astype({
            const.SRC: int,
            const.DST: int,
            const.PRED: float,
            const.Y_TRUE: int,
            const.EVAL_INDEX: int,
            const.SUBREDDIT: str,
            const.RESOURCE: str,
        })

        full = get_eval_df(pred_and_true_df, args)

        full_li += [full]

        y_true_mat = np.array(
            full.groupby(const.EVAL_INDEX)[const.Y_TRUE].apply(
                list).tolist())
        row, col = y_true_mat.nonzero()
        y_true_mat = sp.csr_matrix((y_true_mat[row, col], (row, col)),
                                   shape=y_true_mat.shape)
        assert y_true_mat.nnz == len(pos_out)
        y_true_mat_li.append(y_true_mat)

        pred_and_true_df_li += [pred_and_true_df]

        aps.append(average_precision_score(y_true, y_pred))

        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    # [Summary] Start: Eval using K Negative Samples

    """
    Sample 1 positive item per user, plus K negatives
    """
    if args.verbose:
        print(f"[{split}] Epoch {epoch}")

    full = pd.concat(full_li).reset_index(drop=True)

    get_ranking_results(full, dataset, eval_collection, epoch, split,
                        args, writer)


t0 = time()


def print_time():
    if args.verbose:
        print(f"\t[Time elapsed] {time() - t0:.2f} second s")


datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.verbose:
    print("Experiment time:", datetime_str)

set_directories_from_args(args)
device = args.device

cache_dir = args.cache_dir

logger = setup_logger(osp.basename(__file__), log_dir=args.log_dir)

dataset = RedditDataset(dataset_name=args.dataset_name, args=args,
                        logger=logger)

memory_dim = time_dim = embedding_dim = args.embedding_dim

print_time()
train_df, val_df, test_df, pos_items_train_val_test = dataset.train_val_test_split()
print_time()

if args.do_static_modeling:

    path = osp.join(cache_dir, "community_influence_graph")
    os.makedirs(path, exist_ok=True)

    cig_train_dataset = CIGDataset(path, df=train_df,
                                   phrase=const.TRAIN,
                                   num_src=dataset.num_resource,
                                   num_dst=dataset.num_subreddit,
                                   args=args,
                                   mappings=dataset.mappings)

    cig_train_loader = DataLoader(cig_train_dataset,
                                  batch_size=args_static_modeling.batch_size,
                                  shuffle=True)

    cig_val_dataset = CIGDataset(path, df=val_df,
                                 phrase=const.VAL,
                                 num_src=dataset.num_resource,
                                 num_dst=dataset.num_subreddit,
                                 args=args,
                                 mappings=dataset.mappings,
                                 )

    cig_val_loader = DataLoader(cig_val_dataset,
                                batch_size=args_static_modeling.batch_size,
                                shuffle=True)

    cig_test_dataset = CIGDataset(path, df=test_df,
                                  phrase=const.TEST,
                                  num_src=dataset.num_resource,
                                  num_dst=dataset.num_subreddit,
                                  args=args,
                                  mappings=dataset.mappings)

    cig_test_loader = DataLoader(cig_test_dataset,
                                 batch_size=args_static_modeling.batch_size,
                                 shuffle=True)

    model_static = GNNModel(hidden_size=args_static_modeling.hidden_size,
                            n_node=dataset.urls_df_ge_k_interactions[
                                const.DST].nunique(), args=args).to(
        args.device)

    optimizer_session = torch.optim.Adam(model_static.parameters(),
                                         lr=args_static_modeling.lr,
                                         weight_decay=args_static_modeling.l2)

    del path

else:
    cig_train_loader = cig_test_loader = None
    model_static = optimizer_session = None

if args.evaluate_on_each_subset:
    dataset.get_cold_start_resources()
print_time()

data = dataset[0]
data = data.to(args.device)

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15,
                                                            test_ratio=0.15)

train_src_set = set(train_data.src.tolist())
val_src_set = set(val_data.src.tolist())
test_src_set = set(test_data.src.tolist())

train_val_overlap = train_src_set & val_src_set
train_test_overlap = train_src_set & test_src_set

if args.verbose:
    print("#URLS:")
    print(
        f"\tTrain:\t{len(train_src_set)}\n\tVal:\t{len(val_src_set)}\n\tTest:\t{len(test_src_set)}")

    # Stats on overlapping nodes between train and val/test:
    print(
        f"\tTrain val overlap: {len(train_val_overlap)} ({len(train_val_overlap) / len(val_src_set) * 100:.2f}% of val)")
    print(
        f"\tTrain test overlap:\t{len(train_test_overlap)} ({len(train_test_overlap) / len(test_src_set) * 100:.2f}% of test)")
    print(f"\tVal test overlap:\t{len(val_src_set & test_src_set)}")

    print("\tVal URL not in train dataset:",
          len(set(val_data.src.tolist()) - set(train_data.src.tolist())))
    print("\tTest URL not in train dataset:",
          len(set(test_data.src.tolist()) - set(train_data.src.tolist())))
    print('\tVal URL test dataset not in train:',
          len((set(val_data.src.tolist()) | set(test_data.src.tolist())) - set(
              train_data.src.tolist())))


def get_pos_items_df(data):
    """Negative sampling for training should rely on no knowledge about the test set"""
    pos_items_df_split = pd.DataFrame({
        args.resource: data.src.tolist(),
        const.SUBREDDIT: data.dst.tolist(),
    })
    pos_items_df_split = pos_items_df_split.groupby(args.resource).agg(list)
    return pos_items_df_split


# This method contains duplicate (repetitive) interactions
pos_items_df_train = get_pos_items_df(train_data)
pos_items_df_val = pos_items_df_test = None
logger.info(
    f'train-test overlap: {len(set(train_data.dst.tolist()) & set(test_data.dst.tolist()))}')

logger.info(
    f'train-val overlap: {len(set(train_data.dst.tolist()) & set(val_data.dst.tolist()))}')

val_collection = {
    "All": defaultdict(dict),
    "Cold": defaultdict(dict),
    "Warm": defaultdict(dict),
    "Cold Video Warm Subreddit": defaultdict(dict),
    "Cold Video Cold Subreddit": defaultdict(dict),
    "Warm Video Warm Subreddit": defaultdict(dict),
    "Warm Video Cold Subreddit": defaultdict(dict),
}

test_collection = {
    "All": defaultdict(dict),
    "Cold": defaultdict(dict),
    "Warm": defaultdict(dict),
    "Cold Video Warm Subreddit": defaultdict(dict),
    "Cold Video Cold Subreddit": defaultdict(dict),
    "Warm Video Warm Subreddit": defaultdict(dict),
    "Warm Video Cold Subreddit": defaultdict(dict),
}

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=args.batch_size)

# Check Information Leaks
assert train_data.t.max() <= val_data.t.min()

if args.do_val:
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size)
    assert val_data.t.max() <= test_data.t.min()

neighbor_loader = LastNeighborLoader(data.num_nodes, size=args.num_neighbors,
                                     device=device)
memory_dim = time_dim = embedding_dim = args.embedding_dim

"""
Dynamic Modeling in Section 3.3 - 3.4 of KDD 2023 Paper
"""

memory = Memory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
    args=args,
    mappings=dataset.mappings,
    dataset=dataset,
).to(args.device)

gnn = GraphAttentionEmbedding(
    args,
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(args.device)

link_pred = LinkPredictor(in_channels=embedding_dim,
                          args=args).to(args.device)

gnn_params = [p for n, p in gnn.named_parameters()]
gnn_params_names = [n for n, p in gnn.named_parameters()]
link_pred_params = [p for n, p in link_pred.named_parameters() if
                    n not in gnn_params_names]
link_pred_params_names = [n for n, p in link_pred.named_parameters() if
                          n not in gnn_params_names]

memory_params = [p for n, p in memory.named_parameters() if
                 n not in set(["pretrained_subreddit_embeddings.weight",
                               "resource_embeds.weight",
                               "channel_embeds.weight"]) | set(
                     gnn_params_names) | set(link_pred_params_names)]

if args.do_static_modeling:

    # Only update the embeddings in the rare case we want it to be learnable
    # Not recommended due to potential dataset leak
    sparse_params_names = ["resource_embeddings.weight",
                           "channel_embeddings.weight"]

    sparse_params = [p for n, p in memory.named_parameters() if
                     n in sparse_params_names]

else:
    sparse_params = []

optimizer_dense = torch.optim.Adam([
    {
        'params': gnn_params,
        'name': 'gnn',
    },
    {
        'params': link_pred_params,
        'name': 'link_pred',
    },
    {
        'params': memory_params,
        'name': 'memory',
    }

], lr=args.lr)
scheduler_dense = LinearLR(optimizer=optimizer_dense,
                           start_factor=0.5,
                           end_factor=1.,
                           total_iters=args.scheduler_total_iters)

if len(sparse_params) > 0:
    optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=args.lr)
    scheduler_sparse = LinearLR(optimizer=optimizer_sparse,
                                start_factor=0.5,
                                end_factor=1.,
                                total_iters=args.scheduler_total_iters)


else:
    optimizer_sparse = scheduler_sparse = None

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

writer = SummaryWriter(args.log_dir)

kwargs = {
    'gnn': gnn,
    'memory': memory,
    'link_pred': link_pred,
    'neighbor_loader': neighbor_loader,
    'optimizer_dense': optimizer_dense,
    'optimizer_sparse': optimizer_sparse,
    'scheduler_dense': scheduler_dense,
    'scheduler_sparse': scheduler_sparse,
    'model_static': model_static,
    'optimizer_session': optimizer_session,
}

start_epoch = 1

if args.load_checkpoint_from_epoch >= 0:
    start_epoch = args.load_checkpoint_from_epoch

neg_items_df_train = neg_items_df_val = neg_items_df_test = None

if args.do_val:
    neg_items_df_val = sample_neg_items(pos_items_df_val,
                                        dataset=dataset,
                                        split=const.VAL,
                                        mode=args.eval_sample_method,
                                        min_dst_idx=min_dst_idx,
                                        max_dst_idx=max_dst_idx)
neg_items_df_test = sample_neg_items(pos_items_df_test, dataset=dataset,
                                     split=const.TEST,
                                     mode=args.eval_sample_method,
                                     min_dst_idx=min_dst_idx,
                                     max_dst_idx=max_dst_idx)

for epoch in range(start_epoch, args.epochs + 1):

    # Redo negative sampling of training/val/test dataset
    if args.train_sample_method != const.RANDOM:
        if epoch == 1 or epoch % args.resample_every == 0:
            print(f"Epoch {epoch}\tResampling negative items")
            neg_items_df_train = sample_neg_items(pos_items_df_train,
                                                  dataset=dataset,
                                                  split=const.TRAIN,
                                                  mode=args.train_sample_method)
    else:
        assert neg_items_df_train is None

    if args.load_checkpoint_from_epoch >= 0 and epoch == args.load_checkpoint_from_epoch:
        # Only for the purpose of resetting the model
        # For epoch 0, we only run the initial part of `train()` to initialize the training pipeline
        train(neg_items_df_train, epoch=0)
        load_model(args, **kwargs)

    # Epoch 0 will directly return
    loss = train(neg_items_df_train, epoch)

    if loss is not None:
        # Evaluate with the randomly initialized model when epoch = 0
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        writer.add_scalar(f"Train/loss", loss, epoch)

    if epoch % args.eval_every == 0:

        if args.do_val:
            with torch.no_grad():
                test(val_loader, epoch, neg_items_df_val, val_collection, split=const.VAL, cig_eval_loader=cig_val_loader)

        with torch.no_grad():
            test(test_loader, epoch, neg_items_df_test, test_collection, split=const.TEST, cig_eval_loader=cig_test_loader)



        # for split, performances in performances_d.items():
        #     print(f"{split.capitalize()}")
        #     log_str = "\t"
        #     count = 0
        #     for metric, value in performances.items():
        #         writer.add_scalar(
        #             f"{split.capitalize()}/{metric.capitalize()}", value,
        #             epoch)
        #
        #         log_str += f"{metric.capitalize()}: {value:.4f}\t"
        #         count += 1
        #         if count % 3 == 0 and args.verbose:
        #             print(log_str)
        #             log_str = "\t"
        #
        #     print(log_str)

    save_results_to_excel(args,
                          results_val=val_collection if args.do_val else None,
                          results_test=test_collection)

    writer.add_scalar(f"Train/lr_dense", scheduler_dense.get_last_lr()[-1],
                      epoch)
    if epoch >= 1:
        scheduler_dense.step()
        if scheduler_sparse is not None:
            scheduler_sparse.step()
            writer.add_scalar(f"Train/lr_sparse",
                              scheduler_sparse.get_last_lr()[-1],
                              epoch)

    if epoch % args.save_model_every == 0 and epoch > 0:
        save_model(epoch, args, **kwargs)

print('Done!')
