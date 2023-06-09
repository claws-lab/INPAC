import itertools
import itertools
import os.path as osp
import pickle
import sys
import warnings
from collections import Counter
from typing import Callable, Optional

from tqdm import tqdm

import const
from utils.utils_data import get_modified_time_of_file

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, TemporalData


class RedditDataset(InMemoryDataset):

    def __init__(
            self,
            dataset_name: str,
            args,
            logger,
            # src, dst, t,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
    ):

        self.dataset_name = dataset_name
        self.root = args.cache_dir
        self.args = args
        self.logger = logger
        self.dataset_path = osp.join(self.args.cache_dir, f'reddit_dataset.pkl')

        get_modified_time_of_file(self.dataset_path)
        reddit_dataset = pickle.load(open(self.dataset_path, 'rb'))

        # `urls_df_ge_k_interactions`: the dataset that only keeps resource (e.g. videos) OR users with >= k interactions
        assert reddit_dataset['urls_df_ge_k_interactions'].index.max() == len(
            reddit_dataset['urls_df_ge_k_interactions']) - 1
        self.mappings = reddit_dataset['mappings']

        reddit_dataset['urls_df_ge_k_interactions'] = self.map_node_to_idx(
            reddit_dataset['urls_df_ge_k_interactions'], self.mappings)

        self.mappings['idx2subreddit'] = {v: k for k, v in self.mappings[
            'subreddit2idx'].items()}

        self.mappings['idx2user'] = {v: k for k, v in
                                     self.mappings['user2idx'].items()}

        self.num_subreddit = len(self.mappings['idx2subreddit'])
        self.num_user = len(self.mappings['idx2user'])

        self.min_timestamp = reddit_dataset['urls_df_ge_k_interactions'][
            const.TIMESTAMP].min()

        self.urls_df_ge_k_interactions = reddit_dataset[
            'urls_df_ge_k_interactions']

        self.urls_df_ge_k_interactions[
            const.TIMESTAMP] -= self.min_timestamp

        self.num_resource = self.urls_df_ge_k_interactions[
            args.resource].nunique()
        self.num_user = self.urls_df_ge_k_interactions[
            args.author_identifier].nunique()

        self.padding_of_dst = self.num_resource

        self.num_edges = len(self.urls_df_ge_k_interactions)

        if 'pos_items_df' in reddit_dataset:
            self.pos_items_df = reddit_dataset['pos_items_df']
            self.pos_items_df.sort_index(inplace=True)
            self.pos_items_df_original = reddit_dataset[
                'pos_items_df_original']

        self.mappings['idx2resource'] = {v: k for k, v in
                                         self.mappings[
                                             'resource2idx'].items()}

        self.mappings['idx2user'] = {v: k for k, v in
                                     self.mappings[
                                         'user2idx'].items()}

        # Number of source nodes
        if self.args.node_types == "v_subreddit":
            self.num_src = self.num_resource
            self._num_nodes = self.num_resource + self.num_subreddit
        elif self.args.node_types == "author_subreddit":
            self.num_src = self.num_user
            self._num_nodes = self.num_user + self.num_subreddit

        # Number of destination nodes
        self.num_dst = self.num_subreddit

        super().__init__(self.args.cache_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def map_node_to_idx(self, df, mappings):

        if const.SRC in df and const.DST in df:
            return df

        if self.args.node_types == "author_subreddit":
            print("\tMapping author to indices ...")
            df[const.SRC] = df[self.args.author_identifier].map(
                lambda x: mappings['user2idx'][x])


        elif self.args.node_types == "v_subreddit":
            print("\tMapping resource to indices ...")
            df[const.SRC] = df[self.args.resource].map(
                lambda x: mappings['resource2idx'][x])

        df[const.DST] = df[const.SUBREDDIT].apply(
            lambda x: mappings['subreddit2idx'][x])

        return df

    def train_val_test_split(self, start_from_zero_timestamp=False,
                             pad_item=True):
        """

        :param start_from_zero_timestamp: If True, then the timestamps are shifted such that the first timestamp is 0
        :param padding:
        :param pad_item: Pad the index of items by #resources
        :return:
        """

        path = osp.join(self.args.cache_dir, "train_val_test_split.pt")

        if osp.exists(path):
            if self.args.verbose:
                print(f"\tLoading train/val/test df from {path}")
            self.train_df, self.val_df, self.test_df, self.pos_items_train_val_test = torch.load(
                path)

        else:
            print(f"\tGenerating train/val/test df and save to {path}")

            df = self.urls_df_ge_k_interactions

            if pad_item:
                print(
                    f"\tPadding item index by #resources {self.padding_of_dst}")
                if df[const.SRC].min() == \
                        df[const.DST].min() == 0:
                    df[const.DST] += self.padding_of_dst
            assert df[const.DST].min() == \
                   df[const.SRC].max() + 1

            if start_from_zero_timestamp:
                df[const.TIMESTAMP] = \
                    df[const.TIMESTAMP] - \
                    df[const.TIMESTAMP].min()

            trainDataSize = int(
                len(df.timestamp) * (
                        1 - self.args.test_size - self.args.val_size))
            train_val_cutoff = \
                df.timestamp.iloc[trainDataSize]

            train_val_len = int(
                len(df.timestamp) * (
                        1 - self.args.test_size))
            val_test_cutoff = df.timestamp.iloc[
                train_val_len]

            train_df = df[
                df.timestamp <= train_val_cutoff]
            val_df = df[(
                                df.timestamp > train_val_cutoff) & (
                                df.timestamp <= val_test_cutoff)]
            test_df = df[
                df.timestamp > val_test_cutoff].reset_index(
                drop=True)

            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            pos_items_df_train = self.train_df.groupby(
                const.SRC)[const.DST].apply(set).rename(const.TRAIN)

            pos_items_df_val = self.val_df.groupby(
                const.SRC)[const.DST].apply(set).rename(const.VAL)

            pos_items_df_test = self.test_df.groupby(
                const.SRC)[const.DST].apply(set).rename(const.TEST)

            pos_items_train_val_test = pd.concat(
                [pos_items_df_train, pos_items_df_val, pos_items_df_test],
                axis=1)

            pos_items_train_val_test.sort_index(inplace=True)

            assert pos_items_train_val_test.index.min() == 0
            assert pos_items_train_val_test.index.max() == self.padding_of_dst - 1

            # Convert Null entries to empty set
            pos_items_train_val_test[const.TRAIN] = [
                x if isinstance(x, (set, list)) else set() for x in
                pos_items_train_val_test[const.TRAIN]]
            pos_items_train_val_test[const.VAL] = [
                x if isinstance(x, (set, list)) else set() for x in
                pos_items_train_val_test[const.VAL]]
            pos_items_train_val_test[const.TEST] = [
                x if isinstance(x, (set, list)) else set() for x in
                pos_items_train_val_test[const.TEST]]

            # Items in train but not in test. These items will have very high ratings as they are seen in the training phase. So we ignore them during test

            pos_items_train_val_test[
                'train-test'] = pos_items_train_val_test.apply(
                lambda x: x[const.TRAIN] - x[const.TEST], axis=1)

            pos_items_train_val_test[
                'train-val'] = pos_items_train_val_test.apply(
                lambda x: x[const.TRAIN] - x[const.VAL],
                axis=1)

            # Items in test but not in train
            pos_items_train_val_test[
                'test-train'] = pos_items_train_val_test.apply(
                lambda x: x[const.TEST] - x[const.TRAIN], axis=1)

            pos_items_train_val_test[
                'val-train'] = pos_items_train_val_test.apply(
                lambda x: x[const.VAL] - x[const.TRAIN],
                axis=1)

            assert pos_items_train_val_test.isna().sum().sum() == 0

            pos_items_train_val_test.sort_index(inplace=True)

            self.pos_items_train_val_test = pos_items_train_val_test

            torch.save((self.train_df, self.val_df, self.test_df,
                        pos_items_train_val_test), path)

        resource_with_unseen_subreddits_in_test = self.pos_items_train_val_test[
            self.pos_items_train_val_test['test-train'].apply(len) > 0]
        if self.args.verbose:
            self.logger.info(
                f"\t{len(resource_with_unseen_subreddits_in_test)} has unseen subreddits in test.")
        return self.train_df, self.val_df, self.test_df, self.pos_items_train_val_test

    def get_cold_start_resources(self):

        if not hasattr(self, "cold_start_resource_test"):

            if self.args.verbose:
                print("\t[Cold] Generating cold start resources")
            # self.zero_shot_resource_test = self.pos_items_train_val_test[(self.pos_items_train_val_test[const.TRAIN].isna()) & (self.pos_items_train_val_test[const.TEST].notna())]

            pos_items_train_val_test = self.pos_items_train_val_test.fillna(0)

            self.cold_start_resource_test = []
            self.warm_start_resource_test = []
            self.cold_start_resource_val = []
            self.warm_start_resource_val = []
            self.resource_sr_tuples_train = set()

            count_train_subreddit = Counter()
            count_val_subreddit = Counter()
            count_test_subreddit = Counter()
            count_all_subreddit = Counter()

            for src, row in tqdm(pos_items_train_val_test.iterrows(),
                                 desc="Cold start",
                                 total=len(self.pos_items_train_val_test)):
                # Videos with at least 1 interactions in TEST are kept

                if len(row[const.TRAIN]) > 0:
                    if len(row['test-train']) > 0:
                        if len(row[const.TRAIN]) in [1]:
                            self.cold_start_resource_test.append(src)
                        elif len(row[const.TRAIN]) >= 2:
                            self.warm_start_resource_test.append(src)

                    # Videos with at least 1 interactions in VAL are kept
                    if self.args.do_val and len(row['val-train']) > 0:
                        if len(row[const.TRAIN]) in [1]:
                            self.cold_start_resource_val.append(src)
                        elif len(row[const.TRAIN]) >= 2:
                            self.warm_start_resource_val.append(src)

                self.resource_sr_tuples_train.update(
                    list(itertools.product([src], row[const.TRAIN])))

                count_train_subreddit += Counter(row[const.TRAIN])
                count_val_subreddit += Counter(row[const.VAL])
                count_test_subreddit += Counter(row[const.TEST])

                count_all_subreddit += Counter(
                    row[const.TRAIN] | row[const.VAL] | row[const.TEST])

            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.ioff()
            sns.set(font_scale=1.3)

            # Test
            # Change this line to "count_test_subreddit" to only use the test split
            top_subreddits = dict(count_test_subreddit.most_common())
            K = 20

            top_subreddits = pd.Series(top_subreddits).to_frame(
                'num_interacted_videos')
            top_subreddits[const.SUBREDDIT] = [
                self.mappings['idx2subreddit'][x - self.num_resource] for x in
                top_subreddits.index]
            top_subreddits.index.rename('idx_subreddit', inplace=True)

            top_subreddits = top_subreddits.reset_index()
            top_subreddits['pct'] = top_subreddits['num_interacted_videos'] / (
                    len(self.cold_start_resource_test) + len(
                self.warm_start_resource_test)) * 100

            self.top_test_subreddits_series = top_subreddits
            self.topN_test_subreddits = top_subreddits.iloc[
                                        :int(len(top_subreddits) * 0.1)]

            # Val
            top_subreddits = dict(count_val_subreddit.most_common())
            K = 20

            top_subreddits = pd.Series(top_subreddits).to_frame(
                'num_interacted_videos')
            top_subreddits[const.SUBREDDIT] = [
                self.mappings['idx2subreddit'][x - self.num_resource] for x in
                top_subreddits.index]
            top_subreddits.index.rename('idx_subreddit', inplace=True)

            top_subreddits = top_subreddits.reset_index()
            top_subreddits['pct'] = top_subreddits['num_interacted_videos'] / (
                    len(self.cold_start_resource_val) + len(
                self.warm_start_resource_val)) * 100

            self.top_val_subreddits_series = top_subreddits
            self.topN_val_subreddits = top_subreddits.iloc[
                                       :int(len(top_subreddits) * 0.1)]

            if sys.platform in ['Windows']:

                fig, ax = plt.subplots(figsize=(10, 8))

                sns.barplot(top_subreddits[const.SUBREDDIT][:K],
                            top_subreddits['pct'][:K], ax=ax)
                plt.xlabel('Subreddit')
                plt.ylabel('%Test Videos Contained in the Subreddit')

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                if self.args.dataset_name == "large":
                    name = " (Large Dataset)"

                elif self.args.dataset_name == "small":
                    name = " (Small Dataset)"

                plt.title(
                    f"Most Popular Subreddits on dataset {self.args.dataset_name}")
                plt.savefig('top_subreddits.pdf', dpi=300)

        if self.args.verbose:
            print(
                f"\t[Test] #Cold start: Test: {len(self.cold_start_resource_test)}, Val: {len(self.warm_start_resource_test)}")
            if self.args.do_val:
                print(
                    f"\t[Val] #Cold: {len(self.cold_start_resource_val)}; #Warm: {len(self.warm_start_resource_val)}")

        return self.cold_start_resource_test, self.warm_start_resource_test, self.cold_start_resource_val, self.warm_start_resource_val

    def process_user_subreddit_interactions(self):
        src = torch.LongTensor(self.df[const.SRC].values)
        dst = torch.LongTensor(self.df[const.DST].values) + self.num_subreddit
        time_t = torch.LongTensor(
            self.df[const.TIMESTAMP].values)
        # self._num_nodes = torch.concat([src, dst]).unique().shape[0]
        assert self._num_nodes == dst.max() + 1
        msg = torch.empty(len(src), self.args.message_dim)

        nn.init.xavier_uniform_(msg)
        data = TemporalData(src=src, dst=dst, t=time_t, msg=msg)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def process(self):

        if self.args.node_types == "author_subreddit" and self.args.resource == const.ANYTHING:
            self.process_user_subreddit_interactions()
            return

        urls_df_ge_k_interactions = self.urls_df_ge_k_interactions

        assert urls_df_ge_k_interactions.src.values.min() == urls_df_ge_k_interactions.dst.values.min() == 0

        src = torch.LongTensor(urls_df_ge_k_interactions.src.values)

        if self.args.node_types == "author_subreddit":
            dst = torch.LongTensor(
                urls_df_ge_k_interactions.dst.values) + self.num_user

        elif self.args.node_types == "v_subreddit":
            dst = torch.LongTensor(
                urls_df_ge_k_interactions.dst.values) + self.num_resource

        else:
            raise NotImplementedError

        time_t = torch.LongTensor(
            urls_df_ge_k_interactions.timestamp.values)
        # y = torch.ones(urls_df_ge_k_interactions.shape[0], dtype=torch.long)

        interaction_tuples = list(
            zip(urls_df_ge_k_interactions.src, urls_df_ge_k_interactions.dst))
        self.logger.info(f'#Interactions\t{len(interaction_tuples)}')
        self.logger.info(
            f'#Unique subreddit-resource pairs\t{len(set(interaction_tuples))}')
        self.logger.info(
            f'Density\t{len(set(interaction_tuples)) / (src.unique().shape[0] * dst.unique().shape[0])}')

        self._num_nodes = torch.concat([src, dst]).unique().shape[0]
        assert self._num_nodes == dst.max() + 1
        msg = torch.empty(len(src), self.args.message_dim)

        nn.init.xavier_uniform_(msg)
        data = TemporalData(src=src, dst=dst, t=time_t, msg=msg)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'dataset.pt'

    @property
    def num_nodes(self):
        return self._num_nodes

    def map_and_save_df(self, urls_df_ge_k_interactions, resource_name):

        urls_df_ge_k_interactions = urls_df_ge_k_interactions.reset_index(
            drop=True)
        subreddits_li = urls_df_ge_k_interactions.subreddit.unique()

        resources_li = urls_df_ge_k_interactions[resource_name].unique()

        # The "fullname" field from Reddit dataset
        # NOTE: the subreddit and user indices start from the same number
        users_li = urls_df_ge_k_interactions[
            self.args.author_identifier].unique()

        # Each subreddit is a dst node
        resource2idx = {url: idx for idx, url in
                        enumerate(resources_li)}
        subreddit2idx = {subreddit: idx for idx, subreddit
                         in
                         enumerate(subreddits_li)}

        # NOTE: now we DO NOT pad any indices
        user2idx = {user: idx for idx, user
                    in
                    enumerate(users_li)}

        self.logger.info(f"#URLs: {len(resource2idx)}")
        self.logger.info(f"#Subreddits: {len(subreddit2idx)}")
        self.logger.info(f"#Users: {len(user2idx)}")

        # Each url/v is a src node
        # Each post is an edge

        self.logger.info(f"Create mappings ...")

        mappings = {
            'subreddit2idx': subreddit2idx,
            'resource2idx': resource2idx,
            'user2idx': user2idx
        }

        urls_df_ge_k_interactions = self.map_node_to_idx(
            urls_df_ge_k_interactions, mappings)

        reddit_dataset = {
            "urls_df_ge_k_interactions": urls_df_ge_k_interactions,
            "mappings": mappings,
        }

        self.logger.info(f"Saving reddit_dataset ...")

        pickle.dump(reddit_dataset, open(self.dataset_path, 'wb'))

        return reddit_dataset
