from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import const
from arguments import args


def sample_neg_items(pos_items_df_split, dataset, split: str,
                     mode: str = "random", **kwargs):
    """
    In training, all items that are not positive items in the training set can be negative items.
    In test, all items that are not positive items in training+val+test set can be negative items.
    :param pos_items_df_split:
    :param dataset:
    :param split: str. Can be TRAIN. VAL, TEST
    :return:
    """
    min_dst_idx = kwargs['min_dst_idx']
    max_dst_idx = kwargs['max_dst_idx']
    print(f"Negative Sampling on nodes {min_dst_idx} - {max_dst_idx}")

    if split == const.TRAIN:
        pos_items_df = pos_items_df_split


    elif split in [const.VAL, const.TEST]:
        pos_items_df = dataset.pos_items_train_val_test[f"{split}"]


    else:
        raise ValueError(f"Unknown split {split}")

    """
    # The fastest way
    neg_items_df = pd.Series({
        idx_v: np.random.randint(min_dst_idx, max_dst_idx + 1, size=10)
        for idx_v in pos_items_df.index}).to_frame("neg_items")

    # Archived
    neg_items_df = pd.Series({
        idx_v: list(set(random.sample(range(min_dst_idx, max_dst_idx + 1),
                                      args.num_negative_candidates)) - set(
            pos_items_df.loc[idx_v, const.SUBREDDIT]))

        for idx_v in pos_items_df.index}).to_frame("neg_items")
    """

    neg_sampling_ratio = args.train_neg_sampling_ratio if split == const.TRAIN else args.eval_neg_sampling_ratio

    t0 = time()

    if split == const.TRAIN:
        neg_sampling_ratio *= 100

    if mode == const.RANDOM:

        # We sample more negative items than needed, and then filter out the repeated ones
        neg_items_df = pd.Series({
            idx_v: list(set(np.random.randint(min_dst_idx, max_dst_idx + 1,
                                              size=neg_sampling_ratio * 2)))[
                   :neg_sampling_ratio]
            for idx_v in pos_items_df.index}).to_frame("neg_items")

        assert (neg_items_df.neg_items.apply(
            len) == neg_sampling_ratio).all()


    elif mode == const.EXCLUDE_POSITIVE:
        """
        First sample a larger subset of subreddits, then exclude the positive ones.
        """

        neg_items_df = pd.Series({
            idx_v: set(np.random.randint(min_dst_idx, max_dst_idx + 1,
                                         size=int(
                                             neg_sampling_ratio * 2.)))
            for idx_v in dataset.pos_items_train_val_test.index}).to_frame(
            "neg_items")

        # Sanity check: there should not be any overlap between positive and negative items
        # neg_items_df['pos_items'] = pos_items_df[args.resource]

        # The positive items appeared during test time should not be considered as positive for evaluation
        neg_items_df['pos_items'] = dataset.pos_items_train_val_test.apply(lambda row: (row[const.TRAIN] | row[const.VAL] | row[const.TEST]), axis=1)
        neg_items_df['pos_items_exclude_train'] = \
        dataset.pos_items_train_val_test[f"{split}-train"]

        neg_items_df['neg_items'] = neg_items_df.apply(
            lambda x: list(x['neg_items'] - x['pos_items'])[
                      :neg_sampling_ratio], axis=1)

        if args.do_sanity_check:
            for i, row in tqdm(neg_items_df.iterrows(),
                               total=len(neg_items_df),
                               desc=f"[{split.capitalize()}] Check overlap"):

                if len(set(row['neg_items']) & set(
                        row['pos_items'])) > 0 or len(
                    row['neg_items']) < neg_sampling_ratio:
                    print(row)
                    raise ValueError("Overlap found!")

        neg_items_df.drop(columns=['pos_items'], inplace=True)
        # if split in [VAL, TEST]:
        #     neg_items_df['neg_item'] = neg_items_df['neg_items'].apply(
        #         lambda x: random.choice(x))

    else:
        raise ValueError(f"Unknown mode {mode}")
    if args.verbose:
        print(f"[{split}] Neg Sampling: {time() - t0:.2f} secs")

    return neg_items_df
