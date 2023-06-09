"""
2022.11.27
Adopted from KGAT-pytorch
https://github.com/LunaBlack/KGAT-pytorch.git


"""
import math

import pandas as pd
import numpy as np

import const
from const import *


def cal_hit_ratio(full: pd.DataFrame, args, K: int):
    """Hit Ratio @ top_K"""

    if len(full) == 0:
        return -1

    top_k = full[full[const.RANKING] <= K]
    test_in_top_k = top_k[top_k['test_item'] == top_k[
        'item']]  # golden items hit in the top_K items
    # full[(full[const.RANKING'] != 101) & (full['test_item'] == full['item'])]
    hit = len(test_in_top_k) * 1.0 / full[EVAL_INDEX].nunique()
    if K == args.eval_neg_sampling_ratio and args.eval_sample_method == EXCLUDE_POSITIVE and hit > 1.:
        raise ValueError

    return hit


def cal_ndcg(full: pd.DataFrame, K: int):

    if len(full) == 0:
        return -1

    top_k = full[full[const.RANKING] <= K]
    test_in_top_k = top_k[top_k['test_item'] == top_k['item']].copy()
    test_in_top_k[f'NDCG@{K}'] = test_in_top_k[const.RANKING].apply(
        lambda x: 1 / math.log2(1 + x))  # the rank starts from 1
    return test_in_top_k[f'NDCG@{K}'].sum() * 1.0 / full[EVAL_INDEX].nunique()

def recall_at_k(r, k, all_pos_num):
    # if all_pos_num == 0:
    #     return 0
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def cal_recall(full: pd.DataFrame, K: int):
    """Recall @ top_K"""

    top_k = full[full[const.RANKING] <= K]
    test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
    return len(test_in_top_k) * 1.0 / full['test_item'].nunique()


