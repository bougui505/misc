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
# W: matrix of coefficient of shape (M, D)
# G: kernel matrix of size (M, M)

import torch
import numpy as np
import tqdm
from misc.pytorch import torchify
import misc.rotation


def construct_G(Y, beta):
    """
    >>> M = 10
    >>> D = 3
    >>> beta = 0.5
    >>> Y = torch.randn((M, D))
    >>> G = construct_G(Y, beta)
    >>> G.shape == torch.Size([M, M])
    True
    """
    pdist = torch.cdist(Y, Y)
    G = torch.exp(-(1. / (2 * beta**2)) * pdist**2)
    return G


def compute_P(X, Y, W, sigma_sq, w, G):
    """
    >>> M = 10
    >>> N = 12
    >>> D = 3
    >>> beta = 0.5
    >>> w = 0.5
    >>> sigma_sq = 1.
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> G = construct_G(Y, beta)
    >>> W = torch.randn((M, D))
    >>> P = compute_P(X, Y, W, sigma_sq, w, G)
    >>> P.shape == torch.Size([M, N])
    True
    """
    N, D = X.shape
    M, D = Y.shape
    M, D = W.shape
    M, M = G.shape
    GW = G.mm(W)
    cdist = torch.cdist(torch.Tensor.contiguous(Y + GW), X)  # shape (M, N)
    coeff = -1. / (2 * sigma_sq)
    num = torch.exp(coeff * cdist**2)
    pi = torch.tensor(np.pi)
    tmp = num.sum(axis=0)
    den = tmp + (w / (1. - w)) * (2 * pi * sigma_sq)**(D / 2) * M / N
    P = num / den
    return P


def update_W(X, Y, P, G, lambdav, sigma_sq):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> beta = 0.5
    >>> sigma_sq = 1.
    >>> lambdav = 0.5
    >>> w = 0.5
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> W = torch.randn((M, D))
    >>> G = construct_G(Y, beta)
    >>> P = compute_P(X, Y, W, sigma_sq, w, G)
    >>> W = update_W(X, Y, P, G, lambdav, sigma_sq)
    >>> W.shape == torch.Size([M, D])
    True
    """
    #  wc = np.dot(np.linalg.inv(dp*g+lamb*sigma2*np.eye(m)), (px-dp*y)) (see: https://github.com/Hennrik/Coherent-Point-Drift-Python/blob/8a219221b24362d8a69da5d7b7dc949cbf33ed6e/cpd/cpd_nonrigid.py#L56)
    M, N = P.shape
    P1 = P.mm(torch.ones(N, 1))
    dP1 = torch.diag_embed(P1.flatten())
    dP1_inv = torch.inverse(dP1)
    A = G + lambdav * sigma_sq * dP1_inv
    B = dP1_inv.mm(P).mm(X) - Y
    # W = torch.linalg.solve(A, B)
    W = torch.inverse(A).mm(B)
    return W


class Transform(torch.nn.Module):
    """
    >>> M = 10
    >>> D = 3
    >>> beta = 0.5
    >>> Y = torch.randn((M, D))
    >>> G = construct_G(Y, beta)
    >>> transform = Transform(G, D)
    >>> transform(Y).shape == torch.Size([M, D])
    True
    """
    def __init__(self, G, D):
        """
        """
        super().__init__()
        M, M = G.shape
        zeros = torchify.torchify(torch.zeros(M, D))
        self.G = G
        self.W = torch.nn.Parameter(zeros)

    def __call__(self, Y):
        t = Y + self.G.mm(self.W)
        return t


def update_sigma_sq(X, Y, P, G, W):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> W = torch.randn((M, D))
    >>> sigma_sq = 1.
    >>> w = 0.5
    >>> beta = 0.5
    >>> lambdav = 0.5
    >>> G = construct_G(Y, beta)
    >>> P = compute_P(X, Y, W, sigma_sq, w, G)
    >>> W = update_W(X, Y, P, G, lambdav, sigma_sq)
    >>> sigma_sq = update_sigma_sq(X, Y, P, G, W)
    >>> sigma_sq
    tensor(...)
    """
    N, D = X.shape
    M, D = Y.shape
    M, N = P.shape
    N_P = torch.ones(1, M).mm(P).mm(torch.ones(N, 1))
    T = Y + G.mm(W)
    PT1 = P.T.mm(torch.ones(M, 1))
    dPT1 = torch.diag_embed(PT1.flatten())
    PX = P.mm(X)
    P1 = P.mm(torch.ones(N, 1))
    dP1 = torch.diag_embed(P1.flatten())
    coeff = 1. / (N_P * D)
    tr1 = torch.trace(X.T.mm(dPT1).mm(X))
    tr2 = torch.trace(PX.T.mm(T))
    tr3 = torch.trace(T.T.mm(dP1).mm(T))
    sigma_sq = coeff * (tr1 - 2 * tr2 + tr3)
    sigma_sq = torch.abs(torch.squeeze(sigma_sq))
    return sigma_sq


def inititalize_sigma_sq(X, Y):
    N, D = X.shape
    M, D = Y.shape
    cdist = torch.cdist(Y, X)  # shape (M, N)
    sigma_sq = (1. / (D * N * M)) * (cdist**2).sum()
    return sigma_sq


def get_rmsd(coords1, coords2):
    N = coords1.shape[0]
    rmsd = torch.sqrt(((coords2 - coords1)**2).sum() / N)
    return rmsd


def loss(P):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> sigma_sq = 1.
    >>> beta = 0.5
    >>> w = 0.5
    >>> X = torch.randn((N, D))
    >>> Y = torch.randn((M, D))
    >>> W = torch.randn((M, D))
    >>> G = construct_G(Y, beta)
    >>> P = compute_P(X, Y, W, sigma_sq, w, G)
    >>> ll = loss(P)
    >>> ll
    tensor(...)
    """
    M, N = P.shape
    # return torch.log(P.sum(axis=0)).sum()
    return -(P * torch.log(P)).sum()


def fit(X, Y, w=0., beta=2., maxiter=10000, progress=False):
    """
    >>> N = 12
    >>> M = 10
    >>> D = 3
    >>> X = torch.rand((N, D)) * 10.
    >>> R = torchify.torchify(misc.rotation.get_rotation_matrix(3.14/10., 0., 0.))
    >>> R.shape
    torch.Size([3, 3])
    >>> t = torch.tensor([[0., 0., 0.]]).T
    >>> t.shape
    torch.Size([3, 1])
    >>> Y = (R.mm(X[:M, :].T) + t).T
    >>> Y_opt = fit(X, Y, progress=True)
    >>> get_rmsd(X[:M], Y), get_rmsd(X[:M], Y_opt)
    """
    N, D = X.shape
    M, D = Y.shape
    G = construct_G(Y, beta)
    transform = Transform(G, D)
    W = transform.W
    sigma_sq = inititalize_sigma_sq(X, Y)
    optimizer = torch.optim.Adam(transform.parameters())
    # print(G.mean())
    # sigma_sq = 1.
    if progress:
        pbar = tqdm.tqdm(total=maxiter)
    for i in range(maxiter):
        optimizer.zero_grad()
        P = compute_P(X, Y, W, sigma_sq, w, G)
        _ = transform(Y)
        W = transform.W
        ll = loss(P)
        # sigma_sq_prev = sigma_sq
        sigma_sq = update_sigma_sq(X, Y, P, G, W)
        # delta_sigma_sq = sigma_sq - sigma_sq_prev
        ll.backward(retain_graph=False)
        optimizer.step()
        # P = P.detach()
        # W = W.detach()
        # sigma_sq = sigma_sq.detach()
        if progress:
            pbar.set_description(f'σ²={float(sigma_sq):.3g}|ll={float(ll):.3g}')
            pbar.update(1)
    if progress:
        pbar.close()
    Y_opt = Y + G.mm(W)
    return Y_opt


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
