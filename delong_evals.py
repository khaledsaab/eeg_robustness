from typing import Callable
import dotenv
from typing import List, Sequence
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.models.s4.utils import Mlp, SequenceDecoder
from src.utils import get_logger, last_modification_time, extras, print_config

log = get_logger(__name__)

dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
# Delay the evaluation until we have the datamodule
# So we want the resolver to yield the same string.
OmegaConf.register_new_resolver(
    "datamodule", lambda attr: "${datamodule:" + str(attr) + "}"
)

"""
Delong AUC code from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
and https://github.com/PatWalters/comparing_classifiers/blob/master/delong_ci.py
Bootstrap code from https://github.com/PatWalters/comparing_classifiers/blob/master/bootstrap.py
"""

import pandas as pd
import numpy as np
import scipy.stats
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert (
        len(aucs) == 1
    ), "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[
        :, order
    ]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci


def bootstrap_error_estimate(
    pred, truth, method, alpha=0.95, sample_frac=0.5, iterations=10000
):
    """
    Generate a bootstrapped estimate of confidence intervals
    :param pred: list of predicted values
    :param truth: list of experimental values
    :param method: method to evaluate performance, e.g. matthews_corrcoef
    :param method_name: name of the method for the progress bar
    :param alpha: confidence limit (e.g. 0.95 for 95% confidence interval)
    :param sample_frac: fraction to resample for bootstrap confidence interval
    :param iterations: number of iterations for resampling
    :return: lower and upper bounds for confidence intervals
    """
    index_list = range(0, len(pred))
    num_samples = int(len(index_list) * sample_frac)
    stats = []
    for _ in range(0, iterations):
        sample_idx = resample(index_list, n_samples=num_samples)
        pred_sample = [pred[x] for x in sample_idx]
        truth_sample = [truth[x] for x in sample_idx]
        stats.append(method(pred_sample, truth_sample))
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper, p


def print_avg_pval(model1, model2, dataloader, subgroup_string):

    probs1 = []
    probs2 = []
    targets = []

    model1.cuda()
    model2.cuda()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            targets.extend(y.numpy())
            probs1.extend(model1(x.cuda()).softmax(1)[:, 1].detach().cpu().numpy())
            # probs2.extend(model2(x.cuda()).softmax(1)[:,1].detach().cpu().numpy())
            probs2.extend(
                model2(x.cuda())[:, :2].softmax(1)[:, 1].detach().cpu().numpy()
            )

    probs1 = np.array(probs1)
    probs2 = np.array(probs2)
    targets = np.array(targets)

    auc1, ci1 = calc_auc_ci(targets, probs1, alpha=0.95)
    ci1 = (auc1 - ci1)[0]
    auc2, ci2 = calc_auc_ci(targets, probs2, alpha=0.95)
    ci2 = (auc2 - ci2)[0]

    pval = np.exp(delong_roc_test(targets, probs1, probs2))[0][0]

    print(
        f"{subgroup_string}:\n \t\t Model1: {100*auc1:.1f} +- {100*ci1:.1f} \t Model2: {100*auc2:.1f} +- {100*ci2:.1f} \t p-val: {pval}"
    )

    return None


def train(config: DictConfig):
    """

    Outline training code adapted from https://github.com/HazyResearch/zoo

    Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    """

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(config, False)

    seed_everything(config.seed, workers=True)

    # Init lightning model
    model1: LightningModule = hydra.utils.instantiate(
        config.task, cfg=config, _recursive_=False
    )

    model2: LightningModule = hydra.utils.instantiate(
        config.task, cfg=config, _recursive_=False
    )

    datamodule: LightningDataModule = model1._datamodule

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    pretrained_path1 = "/home/ksaab/Documents/eeg_modeling/checkpoints/gold_only_seed400/last.ckpt"
    pretrained_path2 = "/home/ksaab/Documents/eeg_modeling/checkpoints/pretrain_weak_binary_v6/last.ckpt"

    state_dict1 = torch.load(pretrained_path1)["state_dict"]
    model1.load_state_dict(state_dict1)

    state_dict2 = torch.load(pretrained_path2)["state_dict"]
    model2.load_state_dict(state_dict2)

    print_avg_pval(
        model1, model2, datamodule.test_dataloader(), subgroup_string="Overall"
    )

    ## subgroup evaluations

    # evaluate on age subgroups
    datamodule = hydra.utils.instantiate(
        config.datamodule,
        robust_test=lambda x: np.array(x["hospital"]) == "stanford",
        _recursive_=False,
    )
    datamodule.setup()
    print_avg_pval(
        model1, model2, datamodule.test_dataloader(), subgroup_string="Adults"
    )

    datamodule = hydra.utils.instantiate(
        config.datamodule,
        robust_test=lambda x: np.array(x["hospital"]) == "lpch",
        _recursive_=False,
    )
    datamodule.setup()
    print_avg_pval(
        model1, model2, datamodule.test_dataloader(), subgroup_string="Children"
    )



def dictconfig_filter_key(d: DictConfig, fn: Callable) -> DictConfig:
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    # Using d.items_ex(resolve=False) instead of d.items() since we want to keep the
    # ${datamodule:foo} unresolved for now.
    return DictConfig(
        {
            k: dictconfig_filter_key(v, fn) if isinstance(v, DictConfig) else v
            # for k, v in d.items_ex(resolve=False) if fn(k)})
            for k, v in d.items()
            if fn(k)
        }
    )


@hydra.main(config_path="configs/", config_name="config.yaml")
def run(config: DictConfig):
    # Remove config keys that start with '__'. These are meant to be used only in computing
    # other entries in the config.
    config = dictconfig_filter_key(config, lambda k: not k.startswith("__"))

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    run()
