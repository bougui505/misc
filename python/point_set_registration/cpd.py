#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

# CPD: Coherent Point Drift
# See: https://arxiv.org/pdf/0905.2635.pdf

# Notations
# D: dimension of the point sets
# N, M: number of points in the point sets
# X: the first point set (the data points): shape (N, D)
# Y: the second point set (the GMM centroids): shape (M, D)
# T(Y, theta): transformation to apply to Y, where theta is a set of transformation parameters
# I: identity matrix
# R: rotation matrix of shape (D, D)
# t: translation vector of shape (D, 1)
# s: scaling parameter
# P: probabilites for points X of shape (M, N)
# w: weight for the uniform distribution 0<=w<=1
# ones: column vector of all ones (shape: (?, 1))

import numpy as np
import torch
import tqdm
from misc.pytorch import torchify
import misc.rotation


def transform(Y, R, t, s=1.):
    """
    >>> R = torch.eye(3)
    >>> t = torch.zeros((3, 1))
    >>> Y = torch.randn((10, 3))
    >>> out = transform(Y, R, t)
    >>> out.shape
    torch.Size([10, 3])
    """
    return (s * torch.mm(Y, R.T) + t.T)


def compute_P(X, Y, R, t, s, sigma_sq, w):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> R = torch.eye(D)
    >>> t = torch.zeros((D, 1))
    >>> s = 1
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> sigma_sq = 1.
    >>> w = 0.5
    >>> P = compute_P(X, Y, R, t, s, sigma_sq, w)
    >>> P.shape == torch.Size([M, N])
    True
    """
    N, D = X.shape
    M, D = Y.shape
    Y_trans = (s * R.mm(Y.T) + t).T  # transform(Y, R, t, s=s)
    cdist = torch.cdist(Y_trans, X)
    num = torch.exp((-1 / (2 * sigma_sq)) * cdist**2)
    pi = torch.tensor(np.pi)
    if w < 1:
        w_ratio = w / (1 - w)
    else:
        w_ratio = torch.inf
    # tmp0 = num.sum(axis=0)
    tmp = torch.exp(torch.logsumexp((-1 / (2 * sigma_sq)) * cdist**2, dim=0))
    # print(tmp0 - tmp1)
    den = tmp + (2 * pi * sigma_sq)**(D / 2) * w_ratio * (M / N)
    P = num / den
    return P


def M_step(X, Y, P, compute_s=False):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> R = torch.eye(D)
    >>> t = torch.zeros((D, 1))
    >>> s = 1
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> sigma_sq = 1.
    >>> w = 0.5
    >>> P = compute_P(X, Y, R, t, s, sigma_sq, w)
    >>> R, t, s, sigma_sq = M_step(X, Y, P)
    >>> R.shape
    torch.Size([3, 3])
    >>> t.shape
    torch.Size([3, 1])
    >>> s
    tensor(1.)
    >>> sigma_sq
    tensor(...)
    """
    N, D = X.shape
    M, D = Y.shape
    M, N = P.shape
    ones_N = torch.ones((N, 1))
    ones_M = torch.ones((M, 1))
    N_P = ones_M.T.mm(P).mm(ones_N)
    mu_x = (1 / N_P) * X.T.mm(P.T).mm(ones_M)
    mu_y = (1 / N_P) * Y.T.mm(P).mm(ones_N)
    X_hat = X - ones_N.mm(mu_x.T)
    Y_hat = Y - ones_M.mm(mu_y.T)
    A = X_hat.T.mm(P.T).mm(Y_hat)
    U, S, Vh = torch.linalg.svd(A)
    V = Vh.T
    det = torch.linalg.det(U.mm(V.T))
    C = torch.diag_embed(torch.cat((torch.ones(D - 1), torch.tensor([det]))))
    R = U.mm(C).mm(V.T)
    if compute_s:
        num = torch.trace(A.T.mm(R))
        tmp = torch.diag_embed(P.mm(ones_N).flatten())
        den = torch.trace(Y_hat.T.mm(tmp).mm(Y_hat))
        s = num / den
    else:
        s = torch.tensor(1.)
    t = mu_x - s * R.mm(mu_y)
    tmp = X_hat.T.mm(torch.diag_embed(P.T.mm(ones_M).flatten())).mm(X_hat)
    sigma_sq = (1 / (N_P * D)) * (torch.trace(tmp) - s * torch.trace(A.T.mm(R)))
    sigma_sq = torch.clip(sigma_sq, min=0.)
    # sigma_sq = torch.abs(sigma_sq)
    # sigma = torch.squeeze(torch.sqrt(sigma_sq))
    return R, t, s, sigma_sq


def get_rmsd(coords1, coords2):
    N = coords1.shape[0]
    rmsd = torch.sqrt(((coords2 - coords1)**2).sum() / N)
    return rmsd


def inititalize_sigma_sq(X, Y, R, t, s):
    N, D = X.shape
    M, D = Y.shape
    Y_trans = transform(Y, R, t, s)
    cdist = torch.cdist(Y_trans, X)
    sigma_sq = (1 / (D * N * M)) * (cdist**2).sum()
    return sigma_sq


def EMopt(X, Y, w=0., maxiter=5000, optimize_s=True, progress=False):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> X = torch.rand((N, D)) * 10.
    >>> R = torchify.torchify(misc.rotation.get_rotation_matrix(3.14/4., 0., 0.))
    >>> R.shape
    torch.Size([3, 3])
    >>> t = torch.tensor([[1., 1., 1.]]).T
    >>> t.shape
    torch.Size([3, 1])
    >>> Y = transform(X[:M, :], R, t)
    >>> R_opt, t_opt, s_opt, sigma_sq = EMopt(X, Y, optimize_s=False)
    >>> Y_opt = transform(Y, R_opt, t_opt, s_opt)
    >>> get_rmsd(X[:M], Y), get_rmsd(X[:M], Y_opt), sigma_sq
    """
    N, D = X.shape
    M, D = Y.shape
    R = torch.eye(D)
    t = torch.zeros((D, 1))
    s = 1.
    # cdist = torch.cdist(Y, X)
    sigma_sq = inititalize_sigma_sq(X, Y, R, t, s)
    # sigma_sq = (1 / (D * N * M)) * (cdist**2).sum()
    # sigma = torch.sqrt(sigma_sq)
    if progress:
        pbar = tqdm.tqdm(total=maxiter)
    for i in range(maxiter):
        P = compute_P(X, Y, R, t, s, sigma_sq, w)
        # print(i, P.sum(), sigma)
        sigma_sq_prev = sigma_sq
        R, t, s, sigma_sq = M_step(X, Y, P, compute_s=optimize_s)
        delta_sigma_sq = sigma_sq - sigma_sq_prev
        # print(i, sigma)
        if progress:
            pbar.set_description(f'σ²={float(sigma_sq):.3f}|Δσ²={float(delta_sigma_sq):.3f}')
            pbar.update(1)
        if torch.isnan(sigma_sq):
            print('!!! nan !!!')
            sigma_sq = sigma_sq_prev
        if sigma_sq == 0.:
            break
            # print('!!! 000 !!!')
            # sigma_sq = sigma_sq_prev
        if abs(delta_sigma_sq) < 1e-16:
            break
    if progress:
        pbar.close()
    return R, t, s, sigma_sq


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    from pymol import cmd
    from misc.protein import Coordinates
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p1', '--pdb1')
    parser.add_argument('-p2', '--pdb2')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    cmd.load(args.pdb1, 'pdb1')
    cmd.load(args.pdb2, 'pdb2')
    pdb1 = torchify.torchify(cmd.get_coords('pdb1'))
    pdb2 = torchify.torchify(cmd.get_coords('pdb2'))

    R, t, s, sigma_sq = EMopt(pdb1, pdb2, progress=True)
    coords_opt = transform(pdb2, R, t, s).detach().cpu().numpy()
    Coordinates.change(args.pdb2, 'data/out.pdb', coords_opt)
