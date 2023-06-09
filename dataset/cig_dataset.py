import itertools
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

import const


class CIGDataset(InMemoryDataset):
    """
    Dataset for the Community Influence Graph`(CIG) in Section 3.2 of the KDD 2023 paper.
    """

    def __init__(self, root, df, phrase, transform=None, pre_transform=None,
                 **kwargs):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'val', 'test']
        self.phrase = phrase
        self.df = df.copy()

        self.num_dst = kwargs['num_dst']
        self.num_src = kwargs['num_src']
        self.mappings = kwargs.get('mappings')
        self.args = kwargs['args']

        self.df[const.DST] = self.df[const.DST] - self.num_src + 1
        assert self.df[const.DST].min() >= 1
        print(
            f"\tDST: Min: {self.df[const.DST].min()}, Max {self.df[const.DST].max()}")

        self.df['dst_author_time'] = self.df.apply(
            lambda x: (
                x[const.DST], x[self.args.author_identifier],
                x[const.TIMESTAMP]),
            axis=1)

        # Subreddit index should start from 1
        self.all_sr_author_time_sequences = self.df.groupby(const.SRC)[
            'dst_author_time'].apply(list)

        # Only consider the CIGs with at least two interactions, otherwise no meaningful CIG can be constructed
        if self.phrase in [const.TRAIN]:
            self.all_sr_author_time_sequences = {k: v for k, v in
                                                 self.all_sr_author_time_sequences.items()
                                                 if
                                                 len(v) >= 2}

        print(
            f"\t#{self.phrase} CIGs: {len(self.all_sr_author_time_sequences)}")

        # Using precomputed delta_t_thres
        if isinstance(self.args.delta_t_thres, (float, int)):

            pass

        else:
            if any([getattr(self.args, attr) is None for attr in
                    ['c', 'mu', 'sigma']]):
                self._get_delta_t_thres()

            self.args.delta_t_thres = self.args.mu - self.args.c * self.args.sigma

            print(
                f"Using c={self.args.c:.3f}, mu={self.args.mu:.3f}, sigma={self.args.sigma:.3f}")

        print(
            f"Using delta_t_thres={self.args.delta_t_thres}")

        super(CIGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.phrase}.txt"]

    @property
    def processed_file_names(self):
        return [f"{self.phrase}_{self.args.session_split_method}.pt"]

    def download(self):
        pass

    def _get_delta_t_thres(self):

        timestamps_sequential = []
        timestamps_all = []
        timestamps_same_author = []
        timestamps_diff_author = []
        timestamps_same_author_session = []
        timestamps_diff_author_session = []

        for resource, sequences in tqdm(
                self.all_sr_author_time_sequences.items(),
                desc="Time Diff b/w Interactions"):

            u_set = set()

            # Resource-user pairs
            u2timestamps_li_session = defaultdict(list)
            timestamps_li_session_one_video = []

            for sr, u1, t in sequences:
                if u1 in u_set:
                    pass

                else:
                    # Now the user is aware of the video and, for the first time, sharing the video. So s/he can be influenced by previous sharers
                    timestamps_li_session_one_video += [t]

                    u_set.add(u1)
                u2timestamps_li_session[u1] += [t]

            assert len(u2timestamps_li_session) == len(u_set) == len(
                timestamps_li_session_one_video)
            if len(timestamps_li_session_one_video) >= 2:
                for i in range(len(timestamps_li_session_one_video) - 1):
                    timestamps_diff_author_session += [
                        timestamps_li_session_one_video[i + 1] -
                        timestamps_li_session_one_video[i]]

            for u, timestamps_li_one_user in u2timestamps_li_session.items():
                if len(timestamps_li_one_user) >= 2:
                    timestamps_li_one_user.sort()
                    for i in range(len(timestamps_li_one_user) - 1):
                        timestamps_same_author_session += [
                            timestamps_li_one_user[i + 1] -
                            timestamps_li_one_user[i]]

            del u_set, timestamps_li_session_one_video, u2timestamps_li_session

        timestamps_all = np.array(
            sorted(timestamps_all))
        timestamps_all[timestamps_all == 0] = 1

        timestamps_sequential = np.array(sorted(timestamps_sequential))
        timestamps_sequential[timestamps_sequential == 0] = 1

        timestamps_same_author_session = np.array(
            sorted(timestamps_same_author_session))
        timestamps_same_author_session[
            timestamps_same_author_session == 0] = 1

        timestamps_diff_author_session = np.array(
            sorted(timestamps_diff_author_session))
        timestamps_diff_author_session[
            timestamps_diff_author_session == 0] = 1

        timestamps_same_author = np.array(sorted(timestamps_same_author))
        timestamps_same_author[timestamps_same_author == 0] = 1

        timestamps_diff_author = np.array(sorted(timestamps_diff_author))
        timestamps_diff_author[timestamps_diff_author == 0] = 1

        import matplotlib.pyplot as plt

        plt.ioff()
        import seaborn as sns
        sns.set_theme()
        sns.set(font_scale=1.3)
        fig, ax = plt.subplots(figsize=(8, 8))

        """
        Get dataset from the kde line and bins
        https://stackoverflow.com/questions/37374983/get-data-points-from-seaborn-distplot
        """

        same_or_diff_users = {
            "Session Same": "Multiple Shares by the Same User",
            "Session Diff": "1st Share of Different Users",
        }

        for (name, timestamps) in {
            "Session Same": timestamps_same_author_session,
            "Session Diff": timestamps_diff_author_session,
        }.items():

            bins = 100

            y_max = timestamps.max() + 1
            timestamps_for_plot = timestamps[timestamps <= y_max]

            sns.histplot(data=timestamps_for_plot, kde=True,
                         bins=bins,
                         stat='percent', ax=ax, log_scale=True,
                         color='orangered',
                         line_kws={
                             'color': 'blue'
                         })

            line = ax.lines[0]
            x, y = line.get_data()

            ax.set_xlabel(
                f"Distribution of Delta_t (secs), {self.args.dataset_name.capitalize()} Dataset")
            ax.set_ylabel("%Samples in the Dataset")

            same_or_diff = same_or_diff_users[name]

            plt.title(f"{same_or_diff}")
            plt.tight_layout()

            path = osp.join(self.args.cache_dir,
                            f"delta_t_{self.args.dataset_name.capitalize()}_{name}.pdf")

            if self.args.verbose:
                print(f"\t[Visual] Saving to {path}")

            plt.savefig(path, dpi=300)
            plt.cla()

            # Unimodal Gaussian
            def gauss1(x, *p):
                A1, mu1, sigma1 = p
                return A1 * np.exp(
                    -(x - mu1) ** 2 / (2. * sigma1 ** 2))

            # Bimodal Gaussian
            def gauss2(x, *p):
                A1, mu1, sigma1, A2, mu2, sigma2 = p
                return A1 * np.exp(
                    -(x - mu1) ** 2 / (
                            2. * sigma1 ** 2)) + A2 * np.exp(
                    -(x - mu2) ** 2 / (2. * sigma2 ** 2))

            if name in ["Session Diff"]:
                curve_fit_x = np.log10(x)
                curve_fit_y = y

                """
                `p0` is the initial guess for the fitting coefficients. You can adjust it so that the curve converges to a desired value. A better initialization for them will make the optimization algorithm work better.
                """
                p0 = [2.5, 7., 1.]

                from scipy.optimize import curve_fit

                # Fitting bimodal distribution for same users
                coeff, var_matrix = curve_fit(gauss1, curve_fit_x, curve_fit_y,
                                              p0=p0)
                # amplitude1, mean1, sigma1, amplitude2, mean2, sigma2 = coeff
                amplitude1, mean1, sigma1 = coeff

                self.args.mu = mean1
                self.args.sigma = sigma1

        del timestamps_diff_author, timestamps_same_author, timestamps_sequential, timestamps_all

    def process(self):
        from torch_geometric.data import Data

        data_list = []
        max_num_nodes = 0

        num_sessions_with_atleast_3_interactions = 0

        for resource, sr_author_time_sequence in tqdm(
                self.all_sr_author_time_sequences.items(),
                desc="Constructing session graphs"):
            i = 0
            subreddit_sequence = [x[0] for x in sr_author_time_sequence]

            author_sequence = [x[1] for x in sr_author_time_sequence]

            time_sequence = [x[2] for x in sr_author_time_sequence]

            edge_weight = []

            if len(sr_author_time_sequence) >= 3:
                num_sessions_with_atleast_3_interactions += 1

            nodes = {}  # {15: 0, 16: 1, 18: 2, ...} # node_id -> index of node
            senders = []
            receivers = []
            x = []

            if self.args.session_split_method == const.SEQUENTIAL:

                if len(sr_author_time_sequence) >= 2:
                    for node in subreddit_sequence[:-1]:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        senders.append(nodes[node])  # node_id -> index of node
                    receivers = senders[:]
                    del senders[-1]  # the last item is a receiver
                    del receivers[0]  # the first item is a sender
                else:
                    senders = []
                    receivers = []

            elif self.args.session_split_method == const.ALL:

                if len(sr_author_time_sequence) >= 2:

                    all_combinations = list(
                        itertools.combinations(subreddit_sequence, 2))

                    for s1, s2 in all_combinations:
                        if s1 not in nodes:
                            nodes[s1] = i
                            x.append([s1])
                            i += 1
                        senders.append(nodes[s1])  # node_id -> index of node
                        if s2 not in nodes:
                            nodes[s2] = i
                            x.append([s2])
                            i += 1
                        receivers.append(nodes[s2])

                    assert len(senders) == len(receivers) == len(
                        all_combinations)

                else:
                    senders = []
                    receivers = []



            elif self.args.session_split_method == const.SESSION:

                # Users that have propagated this video ("resource")
                u_set = set()
                # Video-user pairs
                u2sr_t_li = defaultdict(list)

                edge_weights_d = defaultdict(int)

                for sr1, u1, t1 in sr_author_time_sequence:
                    if sr1 not in nodes:
                        nodes[sr1] = i
                        x += [[sr1]]
                        i += 1

                    if u1 in u_set:
                        # The same user is propagating the video for 2nd/3rd ... time in a new subreddit. There exist mutual influence among the subreddits.

                        # sr0 appears earlier than sr1 in the sequence
                        for sr0, t0 in u2sr_t_li[u1]:
                            edge_weights_d[(sr0, sr1)] += 1
                            edge_weights_d[(sr1, sr0)] += 1



                    else:
                        # A new user is aware of the video and, for the first time, sharing the video. So s/he can be influenced by all previous users who share the video
                        for u2, sr_t_li in u2sr_t_li.items():
                            if u1 == u2:
                                pass

                            for sr0, t0 in sr_t_li:
                                delta_t = t1 - t0

                                # Sufficient time has passed since the previous sharing of the video. There might be an influence from sr0 to sr1
                                if delta_t >= self.args.delta_t_thres:
                                    edge_weights_d[(sr0, sr1)] += 1

                    u_set.add(u1)
                    u2sr_t_li[u1] += [(sr1, t1)]

                for (s, r), w in edge_weights_d.items():
                    senders.append(nodes[s])
                    receivers.append(nodes[r])
                    edge_weight += [np.log2(w + 1)]

                del edge_weights_d


            else:
                raise ValueError(
                    f"Invalid session split method {self.args.session_split_method}")

            y = subreddit_sequence[-1]

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)

            x = subreddit_sequence
            x = torch.tensor(x, dtype=torch.long)

            # The last item is the target
            y = torch.tensor([y], dtype=torch.long)

            if edge_weight == []:
                edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)


            else:
                edge_weight = torch.tensor(edge_weight, dtype=torch.float)
                assert edge_weight.size(0) == edge_index.size(1)

            resource = torch.tensor([resource], dtype=torch.long)
            data_list.append(
                Data(x=x, edge_index=edge_index, y=y, resource=resource,
                     edge_weight=edge_weight))

        print(
            f"#Sessions with >=3 interactions: {num_sessions_with_atleast_3_interactions}")

        print(f"Max #Nodes: {max_num_nodes}")

        data, slices = self.collate(data_list)

        assert data.x.min() > 0 and data.x.max() <= self.num_dst

        torch.save((data, slices), self.processed_paths[0])
