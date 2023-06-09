import os
import os.path as osp
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

import const


def get_ranking_results(full: pd.DataFrame, dataset, eval_collection,
                        epoch: int, split: str, args, writer=None):
    full['resource_sr_tuple'] = full.apply(
        lambda x: (x['test_user'], x['test_item']), axis=1)

    if args.evaluate_on_each_subset:
        if split == const.TEST:
            full_filtered = full[~full['resource_sr_tuple'].isin(
                set(dataset.resource_sr_tuples_train))].reset_index(drop=True)
            assert len(full_filtered) % (args.eval_neg_sampling_ratio + 1) == 0

            full_warm = full_filtered[
                full_filtered['test_user'].isin(
                    dataset.warm_start_resource_test)].reset_index(drop=True)
            full_cold = full_filtered[
                full_filtered['test_user'].isin(
                    dataset.cold_start_resource_test)].reset_index(drop=True)

            full_warm_video_warm_subreddit = full_warm[
                full_warm['test_item'].isin(
                    dataset.topN_test_subreddits['idx_subreddit'])].reset_index(
                drop=True)

            full_warm_video_cold_subreddit = full_warm[
                ~full_warm['test_item'].isin(
                    dataset.topN_test_subreddits['idx_subreddit'])].reset_index(
                drop=True)

            full_cold_video_warm_subreddit = full_cold[
                full_cold['test_item'].isin(
                    dataset.topN_test_subreddits['idx_subreddit'])].reset_index(
                drop=True)

            full_cold_video_cold_subreddit = full_cold[
                ~full_cold['test_item'].isin(
                    dataset.topN_test_subreddits['idx_subreddit'])].reset_index(
                drop=True)


        elif split == const.VAL:
            full_filtered = full[~full['resource_sr_tuple'].isin(
                set(dataset.resource_sr_tuples_train))]
            full_warm = full_filtered[
                full_filtered['test_user'].isin(
                    dataset.warm_start_resource_val)]
            full_cold = full_filtered[
                full_filtered['test_user'].isin(
                    dataset.cold_start_resource_val)]

            full_warm_video_warm_subreddit = full_warm[
                full_warm['test_item'].isin(
                    dataset.topN_val_subreddits['idx_subreddit'])]

            full_warm_video_cold_subreddit = full_warm[
                ~full_warm['test_item'].isin(
                    dataset.topN_val_subreddits['idx_subreddit'])]

            full_cold_video_warm_subreddit = full_cold[
                full_cold['test_item'].isin(
                    dataset.topN_val_subreddits['idx_subreddit'])]

            full_cold_video_cold_subreddit = full_cold[
                ~full_cold['test_item'].isin(
                    dataset.topN_val_subreddits['idx_subreddit'])]

            assert len(full_filtered) % (args.eval_neg_sampling_ratio + 1) == 0

        else:
            raise ValueError(f"Unknown split: {split}")

        split_name_and_df = [
            ('All', full),
            ('Warm', full_warm),
            ('Cold', full_cold),
            ("Cold Video Warm Subreddit", full_cold_video_warm_subreddit),
            ("Cold Video Cold Subreddit", full_cold_video_cold_subreddit),
            ("Warm Video Warm Subreddit", full_warm_video_warm_subreddit),
            ("Warm Video Cold Subreddit", full_warm_video_cold_subreddit)
        ]

    else:
        split_name_and_df = [('All', full)]

    for name, full_df in split_name_and_df:

        print(
            f"\n[{split}] {name}: {len(full_df) / (args.eval_neg_sampling_ratio + 1)} entries")
        get_eval_results(full_df, eval_collection[name], epoch, args)

        for metric_name, values in eval_collection[name][epoch].items():
            if isinstance(values, float):
                writer.add_scalar(
                    f'{split.capitalize()}_{name}/{metric_name}',
                    eval_collection[name][epoch][metric_name], epoch)


def get_eval_df(pred_and_true_df: pd.DataFrame, args):
    test = pred_and_true_df[pred_and_true_df.y_true == 1].sort_values(
        const.EVAL_INDEX)[
        [const.SRC, const.DST, const.PRED, const.EVAL_INDEX]].rename({
        const.SRC: 'test_user',
        const.DST: 'test_item',
        const.PRED: 'test_score'
    }, axis=1)

    full_ = pred_and_true_df.sort_values(const.EVAL_INDEX)[
        [const.SRC, const.DST, const.PRED, const.EVAL_INDEX,
         const.Y_TRUE]].rename({
        const.SRC: const.USER,
        const.DST: 'item',
        const.PRED: 'score',
    }, axis=1)

    test = test.reset_index(drop=True)
    full_ = full_.reset_index(drop=True)

    full = pd.merge(full_, test, on=[const.EVAL_INDEX], how='left')

    if args.do_sanity_check:
        entries_y_true_0 = full[
            (full.test_item == full.item) & (full[const.Y_TRUE] == 0)]
        assert len(entries_y_true_0) == 0

    # rank the items according to the scores for each user
    full = full.sample(frac=1, random_state=args.seed)

    full.sort_values([const.EVAL_INDEX, const.SCORE], ascending=[True, False],
                     inplace=True)
    full.reset_index(drop=True, inplace=True)
    # print(full[full.item == full.test_item])

    full[const.RANKING] = full.groupby(const.EVAL_INDEX)[const.SCORE].rank(
        method='first',
        ascending=False)

    full.sort_values([const.EVAL_INDEX, const.RANKING], inplace=True)
    assert (full[const.USER] == full['test_user']).all()
    # full.to_csv('full.csv', index=False)
    return full.reset_index(drop=True)


def get_eval_results(full: pd.DataFrame, eval_dict: defaultdict, epoch: int,
                     args):
    if len(full) == 0:
        return

    from metrics import cal_hit_ratio, cal_ndcg

    y_true_mat = full[const.Y_TRUE].values.reshape(-1, (
            args.eval_neg_sampling_ratio + 1))

    assert (y_true_mat.sum(1) == 1).all()

    results_df_columns = []

    for K in args.Ks:
        hit, ndcg = cal_hit_ratio(full, args, K), cal_ndcg(full, K)
        # self._writer.add_scalar(f'performance/HR{K}', hit, epoch_id)
        # self._writer.add_scalar(f'performance/NDCG{K}', ndcg, epoch_id)

        rec = y_true_mat[:, :K].sum(axis=1)

        rec = rec.mean()

        y_true_series = full.groupby(const.EVAL_INDEX)[const.Y_TRUE].apply(list)
        scores_series = full.groupby(const.EVAL_INDEX)[const.SCORE].apply(list)

        # Sanity check with sklearn

        if args.do_sanity_check:
            ndcg_all = 0
            for eval_index in y_true_series.index:
                ndcg_one_sample = ndcg_score([y_true_series[eval_index]],
                                             [scores_series[eval_index]], k=K)

                ndcg_all += ndcg_one_sample

            ndcg_all /= len(y_true_series)

            print(ndcg, ndcg_all)

        # eval_dict[epoch][f"Hit@{K}"] = hit

        if K in [3, 5, 10, 20]:
            eval_dict[epoch][f"NDCG@{K}"] = ndcg
            eval_dict[epoch][f"Rec@{K}"] = rec
            # print(f'\tHIT@{K}: {hit:.4f}\tNDCG@{K}: {ndcg:.4f}\tRec@{K}: {rec:.4f}')
            print(f'\tRec@{K}: {hit:.4f}\tNDCG@{K}: {ndcg:.4f}')

            # results_df_columns += [f"Hit@{K}", f"NDCG@{K}", f"Rec@{K}"]
            results_df_columns += [f"NDCG@{K}", f"Rec@{K}"]

    eval_indices, ranks = y_true_mat.nonzero()

    mrr = (1 / (ranks + 1)).mean()
    eval_dict[epoch][f"MRR"] = mrr
    print(f'\tMRR: {mrr:.4f}')
    results_df_columns += [f"MRR"]

    args.results_df_columns = results_df_columns


def save_results_to_excel(args, **kwargs):
    print(f'\t[Save] Saving {args.model} results to excel...')

    try:
        path = args.results_dir

        os.makedirs(path, exist_ok=True)

        model_name = "ours"


        with pd.ExcelWriter(
                osp.join(path, f"{model_name}.xlsx")) as writer:
            for split in [const.TRAIN, const.VAL, const.TEST]:

                results = kwargs.get(f'results_{split}')

                if results is not None:
                    # name can be "All", "Cold", "Warm"
                    for name, d in results.items():
                        results_df = pd.DataFrame.from_dict(d).transpose()

                        if results_df.shape[1] == 0:
                            continue

                        results_df.columns = args.results_df_columns

                        if results_df is not None:
                            results_df.to_excel(writer,
                                                sheet_name=f"{split}_{name}")
        print('\t[Save] Done!')


    except Exception as e:
        print("\t[Save] Failed to save results saved")
        traceback.print_exc()


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        'recall': np.array(recall),
        'precision': np.array(pre),
        'ndcg': np.array(ndcg)
    }
