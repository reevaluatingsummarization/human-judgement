import numpy as np
import pickle
import math
from tabulate import tabulate
from scipy.stats import kendalltau
from scipy.stats import pearsonr, spearmanr


def get_pickle(file_path):
    with open(file_path, 'rb') as fp:
        x = pickle.load(fp)
    return x


def print_score_ranges(sd):
    metrics_list = get_metrics_list(sd)
    print_list = []
    headers = ["min", "25-perc", "median", "75-perc", "max", "mean"]
    for m in metrics_list:
        scores = [s['scores'][m] for d in sd.values() for s in d['system_summaries'].values()]
        print_list.append([m,
                           np.min(scores),
                           np.percentile(scores, 25),
                           np.median(scores),
                           np.percentile(scores, 75),
                           np.max(scores),
                           np.mean(scores)])
    print(tabulate(print_list, headers=headers, floatfmt=".4f", tablefmt="pretty"))


def get_metrics_list(sd):
    """
    Does each system summary dict have same all_metrics?
    :param sd: scores dict
    :return: list of all_metrics in the scores dict
    """
    metrics_tuple_set = set(
        [tuple(sorted(list(x['scores'].keys())))
         for d in sd.values() for x in d['system_summaries'].values()])
    assert len(metrics_tuple_set) == 1, "all system summary score dicts should have the same set of all_metrics"
    metrics_list = list(list(metrics_tuple_set)[0])
    return metrics_list


def print_ktau_matrix(metrics, percentile, sd, cutoff_metric='bert_f_score', y_type='ktau'):
    high_pvals = 0
    print(metrics)
    for min_j, mx in enumerate(metrics):
        mean_ktaus = np.zeros((len(percentile), len(metrics)))
        for i, perc in enumerate(percentile):
            for j, my in enumerate(metrics):
                if mx == my or j <= min_j:
                    continue
                ktaus_pvals = [get_doc_y_val(isd, mx, my, y_type=y_type, cutoff_metric=cutoff_metric,
                                             percentile=perc)
                               for isd in sd.values()]
                ktaus = []
                for ktau, pval in ktaus_pvals:
                    # ktaus.append(ktau)
                    if pval <= 0.05:
                        ktaus.append(ktau)
                    else:
                        high_pvals += 1
                mean_ktaus[i, j] = np.mean(ktaus)

        print(mean_ktaus)
        print()

    total_ktaus = (len(metrics) / 2) * (len(metrics) - 1) * len(percentile) * len(sd)
    print(f"total {high_pvals}/{total_ktaus} = {high_pvals * 100 / total_ktaus}% values ignored")


def get_pairwise_correlation(scores_data, key_a, key_b, pval_threshold,
                             filter_metric=None, filter_score=None, top=None, return_ktau_d=False):
    '''
    get the kendall's tau (a correlation ranking score) for the metric pair 'key_a' and 'key_b'
    '''
    kendall_scores = []
    num_ignored = 0
    ktau_d = {}
    for doc_id, isd in scores_data.items():  # isd = "i-th scoe dict, for the i-th doc"
        sumdicts_to_iterate = isd['system_summaries'].values()
        if filter_metric:
            sumdicts_to_iterate = [summary_dict for summary_dict in sumdicts_to_iterate
                                   if summary_dict['scores'][filter_metric] > filter_score]

        if top is not None:
            top_key_a = sorted([(summary_dict['scores'][key_a], summary_dict) for summary_dict in sumdicts_to_iterate],
                               key=lambda x: x[0], reverse=True)[:top]
            top_key_b = sorted([(summary_dict['scores'][key_b], summary_dict) for summary_dict in sumdicts_to_iterate],
                               key=lambda x: x[0], reverse=True)[:top]
            sumdicts_to_iterate = top_key_a + top_key_b
            sumdicts_to_iterate = [summary_dict[1] for summary_dict in sumdicts_to_iterate]

        key_a_scores = [summary_dict['scores'][key_a] for summary_dict in sumdicts_to_iterate]
        key_b_scores = [summary_dict['scores'][key_b] for summary_dict in sumdicts_to_iterate]

        # kendalltau is not reliable if we have less than 4 ranks to compare
        if (len(key_a_scores) < 4) or (len(key_b_scores) < 4):
            num_ignored += 1
            continue

        ktau, pval = kendalltau(key_a_scores, key_b_scores, nan_policy="raise")
        if pval > pval_threshold or math.isnan(ktau):
            num_ignored += 1
            continue

        kendall_scores.append(ktau)
        if doc_id % 100 == 0:
            print(f"done {doc_id}/{len(scores_data)}", end="\r")
        if return_ktau_d:
            ktau_d[doc_id] = ktau
    if return_ktau_d:
        return np.mean(kendall_scores), num_ignored, ktau_d
    else:
        return np.mean(kendall_scores), num_ignored


def filter_summaries(isd, cutoff_metric, percentile):
    if cutoff_metric == 'nas':
        for sys_name in isd['system_summaries']:
            scores = isd['system_summaries'][sys_name]['scores']
            if 'nas' in scores.keys():
                break
            keys = [key for key in scores.keys() if 'recall' in key] + ['mover_score']
            # print(keys)
            scores['nas'] = np.mean([isd['system_summaries'][sys_name]['normed_scores'][m] for m in keys])

    c_scores = [summdict['scores'][cutoff_metric] for summdict in isd['system_summaries'].values()]
    cutoff_score_min = np.percentile(c_scores, percentile[0])
    cutoff_score_max = np.percentile(c_scores, percentile[1])
    filtered_sumdicts_l = [summdict for summdict in isd['system_summaries'].values()
                           if (
                                   (summdict['scores'][cutoff_metric] >= cutoff_score_min)
                                   and (summdict['scores'][cutoff_metric] <= cutoff_score_max)
                           )
                           ]
    return filtered_sumdicts_l


doc_y_types = ['ktau', 'pearson', 'spearman', 'm']


def get_doc_y_val(isd, m1, m2, y_type, cutoff_metric=None, percentile=None):
    assert (y_type in doc_y_types)

    if y_type == 'm':
        return isd['mean_scores'][m1], 0

    filtered_summaries = isd['system_summaries'].values()
    if cutoff_metric is not None:
        filtered_summaries = filter_summaries(isd, cutoff_metric, percentile)

    m1_scores = [summdict['scores'][m1] for summdict in filtered_summaries]
    m2_scores = [summdict['scores'][m2] for summdict in filtered_summaries]

    if y_type == 'ktau':
        ktau, pval = kendalltau(m1_scores, m2_scores, nan_policy="raise")
        if np.isnan(ktau):
            # return high pvalue to ignore
            return 0, 1
        return ktau, pval

    elif y_type == 'pearson':
        pearson_corr, pval = pearsonr(m1_scores, m2_scores)
        return pearson_corr, pval

    elif y_type == 'spearman':
        spearman_corr, pval = spearmanr(m1_scores, m2_scores)
        return spearman_corr, pval
