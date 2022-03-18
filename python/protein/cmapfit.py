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

import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt


def torchify(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def get_dmat(coords, standardize=False):
    """
    >>> coords = torch.randn((10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([10, 10])
    """
    dmat = torch.cdist(coords, coords)
    if standardize:
        dmat = standardize_dmat(dmat)
    return dmat


def addbatchchannel(dmat):
    """
    >>> coords = torch.randn((10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([10, 10])
    >>> dmat = addbatchchannel(dmat)
    >>> dmat.shape
    torch.Size([1, 1, 10, 10])
    """
    if dmat.ndim == 2:
        dmat = torch.unsqueeze(dmat, 0)
        dmat = torch.unsqueeze(dmat, 1)
    return dmat


def get_cmap(coords):
    dmat = get_dmat(coords)
    dmat -= dmat.mean()
    cmap = torch.nn.functional.sigmoid(dmat)
    return cmap


def standardize_dmat(dmat):
    mu = dmat.mean()
    sigma = dmat.std()
    return (dmat - mu) / sigma


def templatematching(dmat, dmat_ref):
    """
    See: https://github.com/hirune924/TemplateMatching/blob/master/Template%20Matching%20(PyTorch%20implementation).ipynb

    >>> coords_ref = torch.randn((10, 3)) * 10.
    >>> coords = coords_ref[4:]
    >>> dmat_ref = get_dmat(coords_ref, standardize=True)
    >>> dmat = get_dmat(coords, standardize=True)

    # >>> p = plt.matshow(dmat.numpy().squeeze())
    # >>> plt.savefig('test_dmat.png')
    # >>> p = plt.matshow(dmat_ref.numpy().squeeze())
    # >>> plt.savefig('test_dmat_ref.png')

    >>> conv = templatematching(dmat, dmat_ref)

    >>> conv.shape
    torch.Size([5, 5])

    # >>> p = plt.matshow(conv.numpy().squeeze())
    # >>> plt.savefig('test_templatematching.png')
    """
    dmat = addbatchchannel(dmat)
    dmat_ref = addbatchchannel(dmat_ref)
    result1 = torch.nn.functional.conv2d(dmat_ref, dmat, bias=None, stride=1, padding=0)
    result2 = torch.sqrt(
        torch.sum(dmat**2) *
        torch.nn.functional.conv2d(dmat_ref**2, torch.ones_like(dmat), bias=None, stride=1, padding=0))
    return (result1 / result2).squeeze(0).squeeze(0)


def get_offset(dmat, dmat_ref):
    """
    >>> coords_ref = torch.randn((10, 3)) * 10.
    >>> ind = np.random.choice(len(coords_ref) - 2)
    >>> coords = coords_ref[ind:]
    >>> dmat_ref = get_dmat(coords_ref, standardize=True)
    >>> dmat = get_dmat(coords, standardize=True)
    >>> offset = get_offset(dmat, dmat_ref)
    >>> offset == ind
    tensor(True)
    """
    conv = templatematching(dmat, dmat_ref)
    diag = torch.diagonal(conv, 0)
    offset = diag.argmax()
    return offset


# def get_offset(dmat, dmat_ref):
#     """
#     >>> coords_ref = torch.randn((10, 3))
#     >>> coords = coords_ref[3:]
#     >>> dmat_ref = get_dmat(coords_ref)
#     >>> dmat = get_dmat(coords)
#     >>> offset = get_offset(dmat, dmat_ref)
#     >>> offset
#     """
#     conv = crosscorrdmat(dmat, dmat_ref)
#     p = len(dmat_ref)
#     n = (1 + np.sqrt(1 + 8 * p)) / 2
#     assert n == int(n)
#     n = int(n)
#     ind = conv.argmax()
#     offsets = list(itertools.combinations(range(n), 2))
#     offset = offsets[ind]
#     return offset

if __name__ == '__main__':
    from pymol import cmd
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--pdb1')
    parser.add_argument('--pdb2')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmd.load(args.pdb1, 'pdb1')
    cmd.load(args.pdb2, 'pdb2')
    pdb1 = torchify(cmd.get_coords('pdb1 and polymer.protein and name CA'))
    pdb2 = torchify(cmd.get_coords('pdb2 and polymer.protein and name CA'))
