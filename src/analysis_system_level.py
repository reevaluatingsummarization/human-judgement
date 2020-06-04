import random
import copy
import numpy as np
from collections import defaultdict as ddict
from scipy.stats import pearsonr, spearmanr, kendalltau


def get_system_level_scores(sd, metrics, agg='mean', nas=False):
    systems = ddict(lambda: ddict(list))

    for isd in sd.values():
        for system_name, scores in isd['system_summaries'].items():
            for m in metrics:
                systems[system_name][m].append(scores['scores'][m])

    for system_name, scores in systems.items():
        for m in scores:
            all_scores = systems[system_name][m]
            if agg == 'mean':
                systems[system_name][m] = np.mean(all_scores)

    if nas:
        min_scores = {}
        max_scores = {}
        for m in metrics:
            min_scores[m] = np.min([systems[sys][m] for sys in systems.keys()])
            max_scores[m] = np.max([systems[sys][m] for sys in systems.keys()])
        for sys in systems:
            systems[sys]['nas'] = np.mean([
                (systems[sys][m] - min_scores[m]) / (max_scores[m] - min_scores[m]) for m in metrics
            ])

    return systems


def get_topk(systems, k, metric='rouge_2_f_score'):
    systems_l = [(name, score[metric]) for name, score in systems.items()]
    systems_l = sorted(systems_l, key=lambda x: x[1], reverse=True)
    topk_system_names = [tup[0] for tup in systems_l[:k]]
    return {name: systems[name] for name in topk_system_names}


def get_correlation(topk_systems, metric_pairs, method='pearson'):
    # disagreement between every pair of metrics for the topk
    corr = {}
    pval = {}
    for pair in metric_pairs:
        m1_scores = []
        m2_scores = []
        for scores in topk_systems.values():
            m1_scores.append(scores[pair[0]])
            m2_scores.append(scores[pair[1]])

        if method == 'pearson':
            correlation, p_value = pearsonr(m1_scores, m2_scores)
        elif method == 'kendalltau':
            correlation, p_value = kendalltau(m1_scores, m2_scores)
        else:
            raise ValueError(f"method {method} not supported")

        key = '_'.join(pair)
        corr[key] = correlation
        pval[key] = p_value

    return corr, pval


def add_synthetic_systems(sd, n, top_only=False):
    """
    :param sd: scores dict
    :param n: number of synthetic systems to add
    :return: sd with added systems
    """
    sd_new = copy.deepcopy(sd)
    for i in range(n):
        name = f'synth_{i}'
        for doc_id in sd_new:
            all_summs = list(sd[doc_id]['system_summaries'].values())
            if top_only:
                possible_summs = list(sorted(all_summs, key=lambda x: x['scores']['litepyramid_recall'], reverse=True))[
                                 :5]
            else:
                possible_summs = all_summs
            random_summary = random.choice(possible_summs)
            sd_new[doc_id]['system_summaries'][name] = random_summary

    return sd_new
