import platform

YOUTUBE_VIDEO_NETLOCS = ['www.youtube.com', 'youtu.be', 'youtube.com',
                         'm.youtube.com']
# Types of YouTube links

CHANNEL = 'channel'
VIDEO = 'video'
ABOUT = 'about'
EMBED = 'embed'

# Short videos
SHORTS = 'shorts'

# These are also videos
CLIP = 'clip'
W = 'w'

# Names for YouTube links
GENERIC = 'generic'
PLAYLIST = 'playlist'
HASHTAG = 'hashtag'
POLICY = 'policy'
POST = 'post'
FEED = 'feed'
RESULTS = "results"
UNKNOWN = 'unknown'
USER = 'user'
WATCH = 'watch'

PATH_SEGMENTS_VIDEOS = [VIDEO, WATCH]

PATH_SEGMENTS_CHANNEL = [CHANNEL, "c"]

# Features in Subreddit
AUTHOR = 'author'
AUTHOR_FULLNAME = 'author_fullname'
CREATED_UTC = 'created_utc'
ID = 'id'
MAX_TIMESTAMP = 'max_timestamp'
MIN_TIMESTAMP = 'min_timestamp'
NETLOC = 'netloc'
POL_LEAN = 'pol_lean'
POST_ID = 'post_id'
RESOURCE = "resource"
SR_AUTHOR_TIME = "sr_author_time"
SUBREDDIT = 'subreddit'
UPDATED = 'updated'
TIME_DIFF = 'time_diff'
TIMESTAMP = 'timestamp'

# Types of resources
URL = 'url'  # Any types of URLs
V = 'v'  # YouTube Videos
MISINFORMATION = 'misinformation'  # URLs from misinformative domains as from FACTOID
ANYTHING = 'anything'  # Any types of posts

IDX_AUTHOR = 'idx_author'
IDX_INTERACTION = 'idx_interaction'
IDX_RESOURCE = 'idx_resource'
IDX_SUBREDDIT = 'idx_subreddit'
IDX_USER = 'idx_user'
IDX_SNAPSHOT = 'idx_snapshot'

SRC = "src"
DST = "dst"
SRC_RELABEL = "src_relabel"
DST_RELABEL = "dst_relabel"
RELATION = "relation"
RANKING = "ranking"
SCORE = "score"

POS_ITEMS = "pos_items"
NEG_ITEMS = "neg_items"

PRED = "pred"
LABEL = "label"
Y_PRED = "y_pred"
Y_TRUE = "y_true"
EVAL_INDEX = "eval_index"

# Sampling method
RANDOM = 'random'  # In use. Fast but not recommended
EXCLUDE_POSITIVE = 'exclude_positive'  # In use. Slow but recommended
PER_INTERACTION = 'per_interaction'

# Evaluation sampling mode
ALL = 'all'
SAMPLE = 'sample'

# train/val/test split
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# Evaluation metrics
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
ACCURACY = 'accuracy'
NDCG = 'ndcg'
TP = 'tp'
FP = 'fp'
TN = 'tn'
FN = 'fn'

# Visualization
node_type2color = {
    USER: '#ffc501',
    SUBREDDIT: '#bc0e4c',
    URL: '#354f60'
}

SAME = 'same'
DIFF = 'diff'
DELTA_T = 'delta_t'

# Types of loss
BPR = 'bpr'
BCE = 'bce'  # Binary Cross entropy

# Aggregation methods for link prediction
DOT = 'dot'
MUL = 'mul'
ADD = 'add'
CONCAT = 'concat'

TITLE = 'title'
DESCRIPTION = 'description'
SELFTEXT = 'selftext'

# Method names or types of embeddings
APPNP = 'appnp'
GAT = 'gat'
GATEDGRAPHCONV = 'gatedgraphconv'
GCN = 'gcn'
LINE = 'line'
TGN = 'tgn'

UNK = 'unk'

# How to split the interactions into different sessions, i.e. How to define a session
# Specified in args.session_split_method
SEQUENTIAL = 'sequential'
BIMODAL = 'bimodal'
OURS = 'ours'
SESSION = 'session'  # Split the session into multiple sessions

CLAWS_LAB = """
  #####  #          #    #     #  #####  
 #     # #         # #   #  #  # #     # 
 #       #        #   #  #  #  # #       
 #       #       #     # #  #  #  #####  
 #       #       ####### #  #  #       # 
 #     # #       #     # #  #  # #     # 
  #####  ####### #     #  ## ##   #####                                                                         
"""

print(CLAWS_LAB)

REDDIT = 'reddit'

