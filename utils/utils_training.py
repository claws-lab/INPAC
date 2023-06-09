import os
import os.path as osp
import warnings
from datetime import datetime

import numpy as np
import torch

import const

warnings.simplefilter(action='ignore', category=FutureWarning)


def set_directories_from_args(args):
    assert args.dataset_name in ["small", "large"]

    if args.node_types == "v_subreddit":
        experiment_setting = f"{args.node_types}_{args.resource}{args.min_inters_resource}_s{args.min_inters_subreddit}"

    elif args.node_types == "author_subreddit":
        experiment_setting = f"{args.node_types}_{args.author_identifier}{args.min_inters_author}_s{args.min_inters_subreddit}"
    else:
        raise NotImplementedError

    if args.do_banned_subreddit_prediction:
        experiment_setting += "_banned_sr"

    print(f"experiment_setting: {experiment_setting}")

    args.cache_dir = osp.join(args.data_dir, const.REDDIT, args.dataset_name,
                              experiment_setting)

    os.makedirs(args.cache_dir, exist_ok=True)

    args.results_dir = osp.join(args.output_dir, 'results', const.REDDIT,
                                args.dataset_name,
                                experiment_setting)

    os.makedirs(args.results_dir, exist_ok=True)

    args.model_full_dir = osp.join(args.model_dir, const.REDDIT,
                                   args.dataset_name,
                                   experiment_setting)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.model_full_dir, exist_ok=True)

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args.log_dir = osp.join('runs', const.REDDIT, args.dataset_name, args.resource, f"{datetime_str}")

    os.makedirs(args.log_dir, exist_ok=True)

    os.makedirs(args.model_full_dir, exist_ok=True)


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n, d = x.size()
    m = y.size(0)

    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def pairwise_cos_sim(a: torch.Tensor, b: torch.Tensor):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def pairwise_cos_sim(A: torch.Tensor, B: torch.Tensor, device="cuda:0"):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)

    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    from torch.nn.functional import normalize
    A_norm = normalize(A, dim=1)  # Normalize the rows of A
    B_norm = normalize(B, dim=1)  # Normalize the rows of B
    cos_sim = torch.matmul(A_norm,
                           B_norm.t())  # Calculate the cosine similarity
    return cos_sim


def pairwise_cos_sim_batch(a: torch.Tensor, b: torch.Tensor, batch_size=16384,
                           device='cuda:0'):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    res = dot_product_batch(a_norm, b_norm, batch_size, device)

    return res


def dot_product_batch(a: torch.Tensor, b: torch.Tensor, batch_size=16384,
                      device='cuda:0'):
    from tqdm import trange
    res = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float).cpu()

    for i in trange(0, a.shape[0], batch_size, desc="Dot Product"):
        a_batch = a[i:i + batch_size].to(device)
        for j in range(0, b.shape[0], batch_size):
            b_batch = b[j:j + batch_size].to(device)
            res_batch = torch.mm(a_batch, b_batch.transpose(0, 1))
            res[i:i + batch_size, j:j + batch_size] = res_batch.cpu()

    return res

