#%%
# Prepare ridge regressors for a specific dataset.
from typing import List
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from einops import rearrange
import lightning.pytorch as pl
from sklearn.linear_model import Ridge # note default metric is r2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from context_general_bci.contexts import context_registry
from context_general_bci.contexts.context_info import FalconContextInfo
from context_general_bci.config import DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.utils import wandb_query_latest, apply_exponential_filter, generate_lagged_matrix
from context_general_bci.inference import get_run_config
from context_general_bci.dataset import SpikingDataset
from context_general_bci.analyze_utils import simple_unflatten

from scripts.offline_analysis.ridge_utils import get_configured_datasets, fit_dataset_and_eval

import sys
import argparse


PER_DATASET_FIT = False
VARIANCE_WEIGHTED = False

scale_ratio = 1.0
# comparator = 'base_45m_1kh_mse-rdyfvp0w'
comparator = 'base_45m_1kh_mse-b3axdmsm'
data_query = ['P4Lab_85_1$']

comparator = 'base_p2_20-2ight1n5'
data_query = ['P2Lab_2137_2$', 'P2Lab_2137_3$', 'P2Lab_2137_10$']
eval_query = data_query
eval_ratio = 0.5

print(context_registry.query(alias=i) for i in data_query)

dataset, eval_dataset = get_configured_datasets(comparator, data_query, eval_query=eval_query)

def get_r2(dataset, eval_dataset, history=0):
    _, predict, truth = fit_dataset_and_eval(dataset, eval_dataset, history=history)
    return r2_score(truth, predict, multioutput='variance_weighted' if VARIANCE_WEIGHTED else 'uniform_average')

score_per_history = []
history_sweep = np.arange(10, 30, 2) # For M1, noted that 0-12 is worse than 14
for history in history_sweep:
    print(f"History: {history}")
    score = get_r2(dataset, eval_dataset, history=history)
    score_per_history.append(score)

for i, score in enumerate(score_per_history):
    print(f"History: {history_sweep[i]}, R2: {score}")
print(f"Best for {scale_ratio}: {history_sweep[np.argmax(score_per_history)]}, R2: {np.max(score_per_history):.3f}")