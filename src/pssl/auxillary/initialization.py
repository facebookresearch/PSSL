"""
Auxillary functions: initialization

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn


def initialization(module: nn.Module):
    """
    Function to initialize module.

    Experiences to be done to understand the importance of initialization.

    Parameters
    ----------
    module: nn.Module
        Network whose parameters are to be initialized.
    """
    if isinstance(module, nn.Linear):
        # nn.init.normal_(module.weight, mean=0, std=1/5)
        # nn.init.xavier_normal_(module.weight)
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def set_all_seed(seed, fast=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
