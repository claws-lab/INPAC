import argparse
import os.path as osp

import const
from const import *

parser = argparse.ArgumentParser(description="")

parser.add_argument('--author_identifier', type=str,
                    choices=[const.AUTHOR, const.AUTHOR_FULLNAME],
                    default=const.AUTHOR,
                    help="Which fields to identify each user. `author` (the displayed name) is preferred since `author_fullname` is not available before 2021.")

parser.add_argument('--batch_size', type=int, default=256,
                    help="the batch size for models. Note that using a smaller batch size is preferred. Otherwise the performance will suffer.")

parser.add_argument('--c', type=float, default=1.,
                    help="The hyperparameter in the equation $\delta t^{thres} = \mu - c \sigma$.")

parser.add_argument('--mu', type=float, default=None,
                    help="The hyperparameter in the equation $\delta t^{thres} = \mu - c \sigma$.")

parser.add_argument('--sigma', type=float, default=None,
                    help="The hyperparameter in the equation $\delta t^{thres} = \mu - c \sigma$.")

parser.add_argument('--comment', type=str, default="",
                    help="Comment for each run. Useful for identifying each run on Tensorboard")
parser.add_argument('--data_dir', type=str, default="data",
                    help="Location to store the processed dataset")
parser.add_argument('--dataset_name', type=str, help="")
parser.add_argument('--demo', action='store_true', help="do demo")
parser.add_argument('--debug', action='store_true', help="do debugging")
parser.add_argument('--device', type=str, default='cuda:0',
                    help="Device to use. When using multi-gpu, this is the 'master' device where all operations are performed.")
parser.add_argument('--device2', type=str, default='cpu',
                    help="For Multi-GPU training")
parser.add_argument('--do_analysis', action='store_true',
                    help="Whether to perform analysis on the dataset")
parser.add_argument('--do_amp', action='store_true',
                    help="Whether to use AMP for mixed precision training")
parser.add_argument('--do_batch', action='store_true',
                    help="Whether to perform batch training in GCN, or perform batch operations in preprocessing / calculations")
parser.add_argument('--do_banned_subreddit_prediction', action='store_true',
                    help="Whether to perform banned subreddit prediction")
parser.add_argument('--do_crosspost', action='store_true',
                    help="Process crosspost data")

parser.add_argument('--do_filter', action='store_true',
                    help="Do we load the filtered URL DataFrame?")

parser.add_argument('--do_kdd', action='store_true',
                    help="Load the old data for KDD 2023 submission")

parser.add_argument('--do_process_urls', action='store_true',
                    help="Whether to process the URLs. If not specified, we assume the urls_RS_20XX-XX.pkl files are aleady processed. The URLs will be loaded from the processed files.")

parser.add_argument('--do_sanity_check', action='store_true',
                    help="Whether to perform sanity check on the dataset, e.g. overlapping items between train/test. This will make the program run slower.")

parser.add_argument('--do_static_modeling', action='store_true',
                    help="Whether to construct the session graph for subreddits")

parser.add_argument('--do_shard', action='store_true',
                    help="do embedding sharding and split the memory to multiple GPUs")
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--do_val', action='store_true')
parser.add_argument('--do_weighted', action='store_true',
                    help="Construct weighted graph instead of multigraph for each graph snapshot")

parser.add_argument('--dropout', type=float, default=0.1,
                    help="Dropout rate (1 - keep probability).")

parser.add_argument('--delta_t_thres', type=float, default=None,
                    help="The cutoff time for constructing CIG in `Influence Graph Construction` of Section 3.2. You can either specify a value or calculate it by fitting a unimodal distribution. For the small dataset, it is ")

parser.add_argument('--embedding_dim', type=int, default=64,
                    help="the embedding size of model")

parser.add_argument('--embedding_dim_user', type=int, default=32,
                    help="The embedding size for the users")
parser.add_argument('--embedding_dim_resource', type=int, default=32,
                    help="The embedding size for the resource (e.g. video)")

parser.add_argument('--epochs', type=int, default=200,
                    help="Number of epochs to train.")
parser.add_argument('--eval_batch_size', type=int, default=256,
                    help="the batch size for models")
parser.add_argument('--eval_every', type=int, default=20,
                    help="How many epochs to perform evaluation?")
parser.add_argument('--eval_neg_sampling_ratio', type=int, default=100,
                    help="How many negative examples to sample for each positive example in val/test?")

parser.add_argument('--eval_embeds_every', type=int, default=-1,
                    help="How many epochs to evaluate embeddings using polarization?")
parser.add_argument('--evaluate_on_each_subset', type=bool, default=True,
                    help="If True, we will split the evaluation into cold-start and warm-start videos.")
parser.add_argument('--eval_sample_method', type=str,
                    choices=[RANDOM, PER_INTERACTION, EXCLUDE_POSITIVE],
                    default=EXCLUDE_POSITIVE,
                    help="Negative sampling method for evaluation dataset")

parser.add_argument('--full_dataset_name', type=str, default="60_months",
                    help="Name of the full dataset")

parser.add_argument('--gpus', type=str, default="0",
                    help="GPUs to use. If using 4 GPUs, type 0,1,2,3")

parser.add_argument('--generate_glove_embeds_for_videos', action='store_true',
                    help="Generate GloVe embeddings for video titles and descriptions")

parser.add_argument('--i_end', type=int, default=None,
                    help="Index of the end dataset.")

parser.add_argument('--i_start', type=int, default=0,
                    help="Index of the start dataset.")

parser.add_argument('--Ks', type=str, default="[1,3,5,10,20,50,100]",
                    help="K for NDCG@K")
parser.add_argument('--keep_embedding_on_cpu', action='store_true')
parser.add_argument('--load_checkpoint_from_epoch', type=int, default=-1,
                    help="If not 0, we will load the checkpoint of this epoch")
parser.add_argument('--loss', type=str, choices=[BPR, BCE], default=BPR,
                    help="Type of loss function")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--max_seq_length', type=int, default=128,
                    help="Maximum sequence length")
parser.add_argument('--message_dim', type=int, default=0,
                    help="If we consider message embedding in video-subreddit graph, set it to nonzero value")
parser.add_argument('--min_inters_author', type=int, default=1,
                    help="Minimum number of interactions for an author to be included in the dataset")
parser.add_argument('--min_inters_resource', type=int, default=3,
                    help="Minimum number of interactions for a URL/video to be included in the dataset")
parser.add_argument('--min_inters_subreddit', type=int, default=1,
                    help="Minimum number of interactions for a subreddit to be included in the dataset")
parser.add_argument('--min_subreddit_sequence_length', type=int, default=1,
                    help="Number of subreddits in a subreddit sequence to be considered in the GPT-2 model")
parser.add_argument('--model', type=str, default=None, help="Model Name")
parser.add_argument('--model_dir', type=str, default="models", help="")

parser.add_argument('--node_types', type=str,
                    choices=["v_subreddit", "author_subreddit",
                             "author_resource"], default="v_subreddit",
                    help="What types of node to include in the GCN bipartite graph?")

parser.add_argument('--num_negative_candidates', type=int, default=1000,
                    help="How many negative examples to sample for each video during the initial sampling?")
parser.add_argument('--num_neighbors', type=int, default=10,
                    help="Number of neighboring nodes in GNN")
parser.add_argument('--num_resource_prototypes', type=int, default=-1, help="")

parser.add_argument('--num_workers', type=int, default=1,
                    help="Number of workers for multiprocessing")
parser.add_argument('--output_dir', type=str, default="outputs",
                    help="Number of epochs to train.")

parser.add_argument('--resample_every', type=int, default=1,
                    help="Number of epochs to resample training dataset.")
parser.add_argument('--resource', type=str,
                    choices=[const.V, const.URL, const.MISINFORMATION],
                    default=V, help="Which resource we use as the src side")

parser.add_argument('--resource_embedding_dim', type=int, default=128,
                    help="the embedding size for resource (e.g. video / url) and channels")

parser.add_argument('--save_embed_every', type=int, default=10,
                    help="How many epochs to save embeddings for visualization?")

parser.add_argument('--save_model_every', type=int, default=20,
                    help="How many epochs to save the model weights?")

parser.add_argument('--path_resource_embeds', type=str, default=None,
                    help="Path to the resource embeddings")
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument('--session_graph_operator', type=str, default=const.APPNP,
                    choices=[const.APPNP, const.GATEDGRAPHCONV],
                    help="Graph operator in session graph.")
parser.add_argument('--session_split_method', type=str, default=const.SESSION,
                    choices=[const.SEQUENTIAL, const.SESSION, const.ALL],
                    help="Method to split a list of subreddits into different session graphs.")
parser.add_argument('--stats_every', type=int, default=int(10e4),
                    help="How many epochs to perform statistics? Now we temporarily ignore this.")
parser.add_argument('--task', type=str, default="", help="task_name")
parser.add_argument('--test_size', type=float, default=0.15, help="")
parser.add_argument('--train_neg_sampling_ratio', type=int, default=1,
                    help="How many negative examples to sample for each positive example in training?")
parser.add_argument('--train_sample_method', type=str,
                    choices=[RANDOM, PER_INTERACTION, EXCLUDE_POSITIVE],
                    default=RANDOM,
                    help="Negative sampling method for training dataset")

parser.add_argument('--val_size', type=float, default=0.15, help="")
parser.add_argument('--verbose', action='store_true', help="")
parser.add_argument('--video_channel_embed_aggregation_method', type=str,
                    choices=[const.ADD, const.MUL, const.CONCAT],
                    default=const.ADD,
                    help="")

args = parser.parse_args()

"""
Whether to use the author's (displayed name) or author_fullname (t2_*) field as the unique identifier for users. Note: we are not sure if the author field is unique for all users or can be modified by users.
"""

args.Ks = eval(args.Ks)

args.link_prediction_aggregation_method = const.ADD


args.scheduler_total_iters = 4

args.comment = 'INPAC'


args_static_modeling = {
    'batch_size': 512,
    'hidden_size': args.embedding_dim,
    'lr': 0.001,
    'lr_dc': 0.1,
    'lr_dc_step': 3,
    'l2': 1e-05,
    'top_k': 20
}

args_static_modeling = argparse.Namespace(**args_static_modeling)
