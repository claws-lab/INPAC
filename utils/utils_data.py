import errno
import functools
import json
import logging
import os
import os.path as osp
import re
import signal
import traceback
import warnings
from time import time
from typing import Union
from urllib.parse import urlparse, parse_qs

import numpy as np
import pytz

from tqdm import tqdm

import const
from const import *
from const import AUTHOR

import pandas as pd


def check_cwd():
    basename = osp.basename(osp.normpath(os.getcwd()))
    assert basename.lower() == "inpac", "Must run this file from parent directory (misinformation/)"


def get_neg_items_df(urls_df: pd.DataFrame, args, mappings):
    """
    Get negative items for each user
    :param urls_df:
    :param args:
    :return:
    """

    # Get negative items for each user
    urls_df.index.rename(IDX_INTERACTION, inplace=True)

    pos_items_df = urls_df.groupby(IDX_RESOURCE)[
        IDX_SUBREDDIT].apply(
        set).reset_index()  # .reset_index(name=IDX_SUBREDDIT)
    pos_items_df = pos_items_df.rename({
        IDX_SUBREDDIT: 'pos_items'
    }, axis=1)

    t0 = time()
    split = "ALL"

    min_subreddit_id, max_subreddit_id = min(
        mappings["subreddit2idx"].values()), max(
        mappings["subreddit2idx"].values())

    urls_df = urls_df.join(pos_items_df['pos_items'], on=IDX_RESOURCE)

    urls_df['neg_items'] = urls_df['pos_items'].apply(lambda x:
                                                      set(np.random.randint(
                                                          min_subreddit_id,
                                                          max_subreddit_id + 1,
                                                          size=args.neg_sampling_ratio * 2)))

    # neg_items_df = pd.Series({
    #     idx_v: set(np.random.randint(min_subreddit_id, max_subreddit_id + 1,
    #                                  size=args.neg_sampling_ratio * 10))
    #     for idx_v in pos_items_df.index}).to_frame("neg_items")

    if args.verbose:
        print(f"[{split}] Neg Sampling: {time() - t0:.2f} secs")

    # neg_items_df['pos_items'] = pos_items_df[IDX_SUBREDDIT].apply(set)
    urls_df['neg_items'] = urls_df.apply(
        lambda x: list(x['neg_items'] - x['pos_items'])[
                  :args.neg_sampling_ratio], axis=1)

    assert (urls_df['neg_items'].apply(len) == args.neg_sampling_ratio).all()

    urls_df.drop(columns=['pos_items'], inplace=True)

    return urls_df


def clean_bots_from_urls_df(urls_df: pd.DataFrame, resource_name):
    # Filter out bot users. This should already been done when getting the URLs with video tokens. We do it again here just in case.
    bot_users = get_list_of_bot_users(urls_df)

    if len(bot_users) > 0:
        print(f'FOUND {len(bot_users)} bot users!!')
        urls_df = urls_df[~urls_df[AUTHOR].isin(bot_users)]

    if resource_name == V:
        # Resource type: the type of resources for grouping
        urls_df = urls_df.query("resource_type == @VIDEO")

        # Filter out URLs with invalid video IDs
        urls_df = urls_df[urls_df[V].str.len() >= 4]

    # Filter out subreddits that are actually users. This is regarding Reddit users' URL patterns
    assert len(urls_df[urls_df.subreddit.str.startswith('u_')]) == 0
    return urls_df.reset_index(drop=True)


def get_urls_df_with_video_tokens(urls_df: pd.DataFrame):
    urls_df.url = urls_df.url.apply(
        lambda x: re.sub(r'//+', '//', x.strip('/').replace('\\', '')))
    # Filter out URLs that are not from youtube

    logging.info(
        f'Before filtering out non-YouTube URLs, #URLs: {len(urls_df)}')

    # urls_df_copy = urls_df.copy()

    urls_df = urls_df[urls_df.is_youtube_link]
    bot_users = get_list_of_bot_users(urls_df)
    urls_df = urls_df[~urls_df[AUTHOR].isin(bot_users)]

    logging.info(f'After filtering out non-YouTube URLs, #URLs: {len(urls_df)}')

    urls_df['parseresult'] = urls_df.url.apply(lambda x: urlparse(x))

    # The resource each URL is linked to. Can be a video, channel, playlist, etc.
    resource_types_li = []

    # Unique identifiers for each video
    v_li = []

    idx_interactions_li = []

    """
    Examples
    Short: 
        http://www.youtube.com/shorts/9nujS6_8m44
        http://www.youtube.com/shorts/HhEaW-uZqN4
    """

    resource_type2path_segments = {
        PLAYLIST: [PLAYLIST],
        ABOUT: [ABOUT],
        USER: [USER],
        POLICY: ['t'],
        POST: [POST],
        FEED: [FEED],
        RESULTS: [RESULTS],
        HASHTAG: [HASHTAG],
    }

    for idx_interaction, row in tqdm(urls_df.iterrows(), total=urls_df.shape[0],
                                     position=0, leave=True):

        # url = row.url.strip('/')
        url = row.url
        # print(idx_interaction, url)
        parseresult = row.parseresult
        if parseresult.scheme == "":
            parseresult = urlparse('https://' + url)

        # Ignore all URLs that are not videos
        # Such as playlists and channels
        paths = parseresult.path.split("/")

        v, resource_type = None, None

        try:
            if len(paths) <= 1:
                assert url in YOUTUBE_VIDEO_NETLOCS or \
                       url.split("://")[1].split('?')[
                           0] in YOUTUBE_VIDEO_NETLOCS
                resource_type = GENERIC

            elif paths[1] in PATH_SEGMENTS_CHANNEL or len(paths) >= 3 and \
                    paths[2] in ["featured", "videos", "playlists", "join",
                                 "about", ""]:
                # paths[3] can be "featured", "videos", "playlists", "join", "about", ""
                # assert len(paths) == 3 or len(paths) == 4
                resource_type = CHANNEL
            else:
                for rtype, path_segments in resource_type2path_segments.items():
                    if paths[1] in path_segments:
                        resource_type = rtype
                        break

            if resource_type is None:

                if parseresult.netloc == "youtu.be":
                    v = paths[1]
                    resource_type = VIDEO

                elif paths[1] in [CLIP, EMBED, SHORTS, V, W]:
                    # Embedded URLs or short videos
                    assert len(paths) == 3
                    v = paths[2]
                    resource_type = VIDEO


                elif parseresult.netloc in YOUTUBE_VIDEO_NETLOCS:
                    queries = parse_qs(parseresult.query)

                    if 'v' not in queries:

                        if paths[1] in PATH_SEGMENTS_VIDEOS:
                            if paths[1] == WATCH:
                                v = parseresult.query

                            else:
                                v = paths[2]
                            resource_type = VIDEO

                        else:
                            # This is a channel
                            assert len(paths) == 2 or len(paths) == 3 and \
                                   paths[2] in ["live"]
                            resource_type = CHANNEL
                    else:
                        v = queries.get(V)[0]
                        resource_type = VIDEO


                else:
                    print(url)
                    print(parseresult)
                    raise NotImplementedError


        except:
            print(url)
            print(parseresult)
            resource_type = UNKNOWN

        if isinstance(v, str):
            v = v.replace('\\_', '_')
            v = v.split('?')[0]

        # print('\tv:\t', v)
        idx_interactions_li += [idx_interaction]
        resource_types_li += [resource_type]
        v_li += [v]
        del v, resource_type, url

    urls_df = urls_df.iloc[:len(idx_interactions_li)]

    urls_df['idx_interaction'] = idx_interactions_li
    urls_df['resource_type'] = resource_types_li
    urls_df[V] = v_li
    urls_df.sort_values(by=TIMESTAMP, inplace=True)
    urls_df.drop('parseresult', axis=1, inplace=True)
    urls_df.set_index('idx_interaction', inplace=True)

    # urls_df_copy.iloc[:200].join(urls_df[['resource_type', V]], on='idx_interaction')
    return urls_df



def get_list_of_bot_users(urls_df: pd.DataFrame):
    bot_users = set()

    for author in tqdm(urls_df[AUTHOR].unique(), desc='Getting Bots'):
        # infixes = ['coupon', 'discount', 'moderator']
        infixes = ['moderator']
        if any(x in author.lower() for x in infixes):
            bot_users.add(author)
        elif author.endswith('-bot') or author.endswith('Bot'):
            bot_users.add(author)

    return list(bot_users)



def get_modified_time_of_file(path):
    import datetime, pathlib
    model_metadata = pathlib.Path(path).stat()
    mtime = datetime.datetime.fromtimestamp(model_metadata.st_mtime)
    ctime = datetime.datetime.fromtimestamp(model_metadata.st_ctime)
    print(f"\t{osp.basename(path)}: modified {mtime} | created {ctime}")
    return mtime, ctime


class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator