#!/usr/bin/env python3
import numpy as np

from collections import defaultdict
from typing import List, Union
import sklearn.metrics as sk

from .eval import multilabel
from .eval import singlelabel
from ..utils import logging
logger = logging.get_logger("visual_prompt")


class Evaluator():
    """
    An evaluator with below logics:

    1. find which eval module to use.
    2. store the eval results, pretty print it in log file as well.
    """

    def __init__(
        self,
    ) -> None:
        self.results = defaultdict(dict)
        self.iteration = -1
        self.threshold_end = 0.5

    def update_iteration(self, iteration: int) -> None:
        """update iteration info"""
        self.iteration = iteration

    def update_result(self, metric: str, value: Union[float, dict]) -> None:
        if self.iteration > -1:
            key_name = "epoch_" + str(self.iteration)
        else:
            key_name = "final"
        if isinstance(value, float):
            self.results[key_name].update({metric: value})
        else:
            if metric in self.results[key_name]:
                self.results[key_name][metric].update(value)
            else:
                self.results[key_name].update({metric: value})

    def classify(self, probs, targets, test_data, multilabel=False):
        """
        Evaluate classification result.
        Args:
            probs: np.ndarray for num_data x num_class, predicted probabilities
            targets: np.ndarray for multilabel, list of integers for single label
            test_labels:  map test image ids to a list of class labels
        """
        if not targets:
            raise ValueError(
                "When evaluating classification, need at least give targets")

        if multilabel:
            self._eval_multilabel(probs, targets, test_data)
        else:
            self._eval_singlelabel(probs, targets, test_data)

    def _eval_singlelabel(
        self,
        scores: np.ndarray,
        targets: List[int],
        eval_type: str
    ) -> None:
        """
        if number of labels > 2:
            top1 and topk (5 by default) accuracy
        if number of labels == 2:
            top1 and rocauc
        """
        acc_dict = singlelabel.compute_acc_auc(scores, targets)

        log_results = {
            k: np.around(v * 100, decimals=2) for k, v in acc_dict.items()
        }
        save_results = acc_dict

        self.log_and_update(log_results, save_results, eval_type)

    def _eval_multilabel(
        self,
        scores: np.ndarray,
        targets: np.ndarray,
        eval_type: str
    ) -> None:
        num_labels = scores.shape[-1]
        targets = multilabel.multihot(targets, num_labels)

        log_results = {}
        ap, ar, mAP, mAR = multilabel.compute_map(scores, targets)
        f1_dict = multilabel.get_best_f1_scores(
            targets, scores, self.threshold_end)

        log_results["mAP"] = np.around(mAP * 100, decimals=2)
        log_results["mAR"] = np.around(mAR * 100, decimals=2)
        log_results.update({
            k: np.around(v * 100, decimals=2) for k, v in f1_dict.items()})
        save_results = {
            "ap": ap, "ar": ar, "mAP": mAP, "mAR": mAR, "f1": f1_dict
        }
        self.log_and_update(log_results, save_results, eval_type)

    def log_and_update(self, log_results, save_results, eval_type):
        log_str = ""
        for k, result in log_results.items():
            if not isinstance(result, np.ndarray):
                log_str += f"{k}: {result:.2f}\t"
            else:
                log_str += f"{k}: {list(result)}\t"
        logger.info(f"Classification results with {eval_type}: {log_str}")
        # save everything
        self.update_result("classification", {eval_type: save_results})



def get_and_print_results(log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    # print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr)

    return 100*fpr, 100*auroc, 100*aupr


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    # print("auroc aupr fpr", auroc, aupr, fpr)
    return auroc, aupr, fpr

def print_measures(log, auroc, aupr, fpr, recall_level=0.95):
    if log == None:
        print('FPR{:d}:\t{:.2f}\t'.format(int(100 * recall_level), 100 * fpr), 'AUROC: \t{:.2f}\t'.format(100 * auroc), 'AUPR:  \t{:.2f}\t'.format(100 * aupr))
        # print('& {:.2f} '.format(100 * fpr), '& {:.2f} '.format(100 * auroc), '& {:.2f} '.format(100 * aupr))
        # print('AUROC: \t\t{:.2f}'.format(100 * auroc))
        # print('AUPR:  \t\t{:.2f}'.format(100 * aupr))
    else:
        logger.info('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        logger.info('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out