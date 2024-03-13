# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.runner import DistEvalHook as BasicDistEvalHook
from collections import Counter

class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)

class ActionResults(object):
    def __init__(self, name):
        self.name = name
        self.nseq = 0
        self.camDiffs = []
        self.corresN = {}
    
    def csv_row(self):
        corres2 = self.corresN[2] if 2 in self.corresN else -1
        corres3 = self.corresN[3] if 3 in self.corresN else -1
        counts = Counter(list(self.camDiffs))
        cam1 = counts["Camera1"] if "Camera1" in counts else 0
        cam2 = counts["Camera2"] if "Camera2" in counts else 0
        cam3 = counts["Camera3"] if "Camera3" in counts else 0
        return [self.name, corres2, corres3, self.nseq, cam1, cam2, cam3]

def confusion_matrix(y_pred, y_real, normalize=None, num_labels=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set) if num_labels is None else num_labels
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc

def match_array(scores, labels, k=1):
    max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
    return match_array

def top_k_by_action(scores, labels, k=1):
    labels = np.array(labels)[:, np.newaxis]
    max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]

    match_by_action = {}

    for i in range(len(labels)):
        label = labels[i][0]
        pred = max_k_preds[i]

        if label not in match_by_action.keys():
            match_by_action[label] = []
        match_by_action[label].append(np.logical_or.reduce(pred == np.array(label)))

    topk_by_action = {}
    for action in match_by_action.keys():
        topk_by_action[action] = sum(match_by_action[action]) / len(match_by_action[action])

    return topk_by_action

def group_to_action(group):
    return int(group[-3:]) - 1

def clustering_by_action(scores, labels, groups, cams, action_res, ncorres=3, k=1):
    labels = np.array(labels)[:, np.newaxis]
    max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    preds = [pred[0] for pred in max_k_preds]

    pred_by_group = {}

    for pred, label, group, cam in zip(preds, labels, groups, cams):
        label = label[0] # TODO : check if a match option is necessary

        if group not in pred_by_group.keys():
            pred_by_group[group] = []
        pred_by_group[group].append((pred, cam))

    corres_all = []
    corres_by_action = {}
    camdiff_by_action = {}
    for group in pred_by_group.keys():
        if len(pred_by_group[group]) < ncorres:
            print("Group ", group, " has ", len(pred_by_group[group]), " views")
            continue

        action_number = group_to_action(group)
        if action_number not in corres_by_action.keys():
            corres_by_action[action_number] = []
        if action_number not in camdiff_by_action.keys():
            camdiff_by_action[action_number] = []

        onlypreds = [pred[0] for pred in pred_by_group[group]]
        counts = Counter(list(onlypreds))

        enough = any(count >= ncorres for count in list(counts.values()))

        if not enough and ncorres > 2:
            justOneDiff = any(count == ncorres - 1 for count in list(counts.values()))
            if justOneDiff:
                valDiff = next(key for key, count in counts.items() if count == 1)
                idx = onlypreds.index(valDiff)
                camDiff = pred_by_group[group][idx][1]

                camdiff_by_action[action_number].append(camDiff)
                action_res[action_number].camDiffs = camdiff_by_action[action_number]

        corres_by_action[action_number].append(enough)

        corres_all.append(enough)

    for action, corres in corres_by_action.items():
        corres_by_action[action] = sum(corres) / len(corres)
        action_res[action].corresN[ncorres] = sum(corres) / len(corres)

    return corres_by_action, camdiff_by_action
    total_corres = sum(corres_all) / len(corres_all)

    return corres_by_action, total_corres


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        matchs = match_array(scores, labels, k)
        topk_acc_score = matchs.sum() / matchs.shape[0]
        res.append(topk_acc_score)

    return res


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)    


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
