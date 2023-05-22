"""
Regularizing functions.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2023, Vivien Cabannes
"""

from typing import Union

import torch
import torch.nn as nn


def ortho_reg(x: torch.Tensor, centered=False):
    """
    Orthogonal features penalization.

    .. math::
        y = \| E[x x^\top] - I \|^2

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * number of dimension`.
    centered: bool, optional
        Either to center `x` before computing covariance. Default is False.

    Returns
    -------
    loss: torch.Tensor
        Empirical orthogonal regularization loss.
    """
    if centered:
        x = x - x.mean(dim=0)
    x = x.transpose(1, 0) @ x / x.size(0)
    x = x - torch.eye(x.size(0), device=x.device)
    loss = torch.sum(x**2)
    return loss


def ortho_reg_contrastive(x: torch.Tensor, centered=False):
    """
    Orthogonal features penalization implemented to provide unbiased stochastic gradient.

    .. math::
        y = E[(x^\top x')^2 - 2 E[(x^\top x)] + p

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * number of dimension` (`2n * p`).
    centered: bool, optional
        Either to center `x` before computing covariance. Default is False.

    Returns
    -------
    loss: torch.Tensor
        Empirical orthogonal regularization loss.
    """
    if centered:
        x = x - x.mean(dim=0)
    n = x.size(0) // 2
    if n < 1:
        raise ValueError(f"Batch size should be at least two, got {x.size(0)}.")
    # Quadratic part
    y = x[:n] @ x[n:].transpose(1, 0)
    loss = torch.mean(y**2)
    # Linear part
    loss = loss - torch.mean(x**2) * 2 * x.size(1)
    loss = loss + x.size(1)
    return loss


def ortho_reg_contrastive_one_to_one(
    net: nn.Module,
    inputs: torch.Tensor,
    outputs: Union[None, torch.Tensor] = None,
    **kwargs,
):
    """
    Compute unbiased version of VCReg with only one comparison for each output in the two different heads.

    Parameters
    ----------
    net: torch.nn.Module
        Pytorch neural network module.
    inputs: torch.Tensor
        Tensor of inputs to feed the networks of size `n * m * in_dim`,
        where `n` is the number of input samples, and `m` of augmentations per inputs.
    outputs: torch.Tensor, optional
        To avoid recomputing outputs inside function scope. Default is None.

    Returns
    -------
    loss: torch.Tensor
        Empirical average of augmentation differences.
    """
    n, m, in_dim = inputs.shape
    if outputs is None:
        outputs = net(inputs.reshape(-1, in_dim)).reshape(n, m, -1)

    k = outputs.shape[-1]
    d = n // 2
    psi1 = outputs[:d]
    psi2 = outputs[d:]

    ortho_reg = torch.sum(torch.sum(psi1 * psi2, dim=-1) ** 2)
    ortho_reg = ortho_reg - torch.sum(outputs**2)
    ortho_reg = ortho_reg / (d * m) + k
    return ortho_reg
