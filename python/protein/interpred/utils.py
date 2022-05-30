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
from misc import randomgen
from pymol import cmd
import numpy as np
import itertools


def get_dmat(coords):
    """
    >>> coords = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords.shape
    torch.Size([1, 85, 3])

    >>> dmat = get_dmat(coords)
    >>> dmat.shape  # batchsize, channel, n, n
    torch.Size([1, 1, 85, 85])
    """
    dmat = torch.cdist(coords, coords)
    dmat = dmat[:, None, ...]  # Add the channel dimension
    return dmat


def PCA(X, p):
    """
    >>> coords = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords.shape
    torch.Size([1, 85, 3])
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([1, 1, 85, 85])
    >>> dmat = torch.squeeze(dmat)

    # >>> plt.matshow(dmat)
    # >>> plt.show()

    >>> dmat.shape
    torch.Size([85, 85])
    >>> T_a = PCA(dmat, p=8)
    >>> T_a.shape
    torch.Size([85, 8])

    >>> coords = get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA')
    >>> dmat = get_dmat(coords)
    >>> dmat = torch.squeeze(dmat)
    >>> T_b = PCA(dmat, p=8)

    # >>> out = torch.matmul(T_a, T_b.T)
    # >>> plt.matshow(out)
    # >>> plt.colorbar()
    # >>> plt.show()

    """
    n = X.shape[0]
    u = X.mean(axis=0)
    sigma = X.std(axis=0)
    B = (X - u) / sigma**2
    C = (1 / (n - 1)) * torch.matmul(B, B.T)
    Lambda, V = torch.linalg.eigh(C)
    Lambda = torch.flip(Lambda, dims=(0, ))
    V = torch.flip(V, dims=(1, ))
    W = V[:, :p]
    T = torch.matmul(B, W)
    return T


def get_input(coords_a, coords_b, input_size, return_normalizer=False):
    dmat_a = get_dmat(coords_a[None, ...])
    dmat_b = get_dmat(coords_b[None, ...])
    # Normalize the distance matrices
    normalizer = Normalizer([dmat_a, dmat_b])
    dmat_a, dmat_b = normalizer.transform([dmat_a, dmat_b])
    #################################
    dmat_a = torch.nn.functional.interpolate(dmat_a, size=input_size)
    dmat_b = torch.nn.functional.interpolate(dmat_b, size=input_size)
    inp = torch.cat((dmat_a, dmat_b), dim=1)
    if return_normalizer:
        return inp, normalizer
    else:
        return inp


def get_cmap(coords, threshold=8.):
    dmat = get_dmat(coords)
    cmap = (dmat <= threshold)
    cmap = cmap.to(torch.float)
    return cmap


def get_inter_dmat(coords_a, coords_b):
    """
    >>> coords_a = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords_a.shape
    torch.Size([1, 85, 3])
    >>> coords_b = get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA')
    >>> coords_b.shape
    torch.Size([1, 13, 3])
    >>> dmat = get_inter_dmat(coords_a, coords_b)
    >>> dmat.shape
    torch.Size([1, 1, 85, 13])
    """
    dmat = torch.cdist(coords_a, coords_b)
    dmat = dmat[:, None, ...]  # Add the channel dimension
    return dmat


def get_interpred(out_a, out_b):
    """
    >>> out_a = torch.ones(1, 64, 83)
    >>> out_b = torch.ones(1, 64, 78)
    >>> out = get_interpred(out_a, out_b)
    >>> out.shape
    torch.Size([1, 83, 78])
    >>> out.min()
    tensor(1.)
    >>> out.max()
    tensor(1.)
    """
    n_class = out_a.shape[1]
    out = torch.einsum('ijk,lmn->ikn', out_a, out_b) / n_class**2
    return out


def get_inter_cmap(coords_a, coords_b, threshold=8.):
    """
    >>> coords_a = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords_a.shape
    torch.Size([1, 85, 3])
    >>> coords_b = get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA')
    >>> coords_b.shape
    torch.Size([1, 13, 3])
    >>> cmap = get_inter_cmap(coords_a, coords_b)
    >>> cmap.shape
    torch.Size([1, 1, 85, 13])
    >>> cmap
    tensor([[[[0., 0., 0.,  ..., 0., 0., 1.],
              [0., 0., 0.,  ..., 0., 0., 1.],
              [0., 0., 0.,  ..., 0., 0., 0.],
              ...,
              [0., 0., 0.,  ..., 0., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.],
              [0., 0., 0.,  ..., 0., 0., 0.]]]], dtype=torch.float64)
    """
    dmat = get_inter_dmat(coords_a, coords_b)
    cmap = (dmat <= threshold)
    cmap = cmap.to(torch.double)
    return cmap


def get_coords(pdb, selection='polymer.protein', return_seq=False):
    """
    >>> coords = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords.shape
    torch.Size([1, 85, 3])
    >>> coords, seq = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> seq
    'ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV'
    >>> coords, seq = get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA', return_seq=True)
    >>> seq
    'ETFSDLWKLLPEN'
    """
    pymolname = randomgen.randomstring()
    cmd.load(pdb, pymolname)
    # Remove alternate locations (see: https://pymol.org/dokuwiki/doku.php?id=concept:alt)
    cmd.remove("not alt ''+A")
    cmd.alter(pymolname, "alt=''")
    ######################################################################################
    coords = cmd.get_coords(f'{pymolname} and {selection}')
    coords = coords[None, ...]  # Add the batch dimension
    coords = torch.tensor(coords)
    if not return_seq:
        cmd.delete(pymolname)
        return coords
    else:
        seq = get_seq(pymolname, selection)
        cmd.delete(pymolname)
        return coords, seq


def get_seq(pymolname, selection):
    seq = cmd.get_fastastr(f'{pymolname} and {selection} and present')
    seq = seq.split()[1:]
    seq = ''.join(seq)
    seq = seq.upper()
    return seq


def encode_seq(seq):
    """
    >>> coords, seq = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> coords.shape
    torch.Size([1, 85, 3])
    >>> len(seq)
    85
    >>> onehot = encode_seq(seq)
    >>> onehot.shape
    torch.Size([85, 21])
    >>> (onehot.sum(axis=1) == 1.).all()
    tensor(True)
    """
    mapping = dict(zip('RHKDESTNQCGPAVILMFYW', range(20)))
    n = len(seq)
    onehot = torch.zeros((n, 21))
    for i, aa in enumerate(seq):
        if aa in mapping:
            j = mapping[aa]
        else:
            j = 20  # (Non-standard residue)
        onehot[i, j] = 1
    return onehot


def get_cmap_seq(coords, seq, threshold=8.):
    """
    >>> coords, seq = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> coords.shape
    torch.Size([1, 85, 3])
    >>> cmap_seq = get_cmap_seq(coords, seq)
    >>> cmap_seq.shape
    torch.Size([1, 21, 85, 85])
    >>> cmap_seq.sum(axis=1)
    tensor([[[2., 2., 2.,  ..., 0., 0., 0.],
             [2., 2., 2.,  ..., 0., 0., 0.],
             [2., 2., 2.,  ..., 0., 0., 2.],
             ...,
             [0., 0., 0.,  ..., 2., 2., 2.],
             [0., 0., 0.,  ..., 2., 2., 2.],
             [0., 0., 2.,  ..., 2., 2., 2.]]])
    """
    interseq = get_inter_seq(seq, seq)
    # interseq.shape: torch.Size([1, 42, 85, 85])
    interseq = interseq[:, :21, ...] + interseq[:, 21:, ...]
    # interseq.shape: torch.Size([1, 21, 85, 85])
    cmap = get_cmap(coords, threshold)
    # cmap.shape: torch.Size([1, 1, 85, 85])
    cmap_seq = cmap * interseq
    return cmap_seq


def get_inter_seq(seq_a, seq_b):
    """
    >>> coords_a, seq_a = get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> coords_a.shape
    torch.Size([1, 85, 3])
    >>> len(seq_a)
    85
    >>> coords_b, seq_b = get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA', return_seq=True)
    >>> coords_b.shape
    torch.Size([1, 13, 3])
    >>> len(seq_b)
    13
    >>> interseq = get_inter_seq(seq_a, seq_b)
    >>> interseq.shape
    torch.Size([1, 42, 85, 13])
    >>> onehot_a = encode_seq(seq_a)
    >>> onehot_b = encode_seq(seq_b)
    >>> (interseq[0, :21, :, 0].T == onehot_a).all()
    tensor(True)
    >>> (interseq[0, 21:, 0, :].T == onehot_b).all()
    tensor(True)
    """
    na = len(seq_a)
    nb = len(seq_b)
    onehot_a = encode_seq(list(seq_a))
    onehot_b = encode_seq(list(seq_b))
    idx_pairs = torch.cartesian_prod(torch.arange(na), torch.arange(nb))
    cartprod = torch.concat((onehot_a[idx_pairs[:, 0]], onehot_b[idx_pairs[:, 1]]), axis=1)
    cartprod = cartprod.reshape((na, nb, 21 * 2))
    cartprod = cartprod.moveaxis(-1, 0)
    return cartprod[None, ...]  # Add the batch dimension


class Normalizer(object):
    def __init__(self, batch):
        """
        >>> batch = [1 + torch.randn(1, 1, 249, 249), 2 + 2* torch.randn(1, 1, 639, 639), 3 + 3 * torch.randn(1, 1, 390, 390), 4 + 4 * torch.randn(1, 1, 131, 131)]
        >>> normalizer = Normalizer(batch)
        >>> [torch.round(e) for e in normalizer.mu]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> [torch.round(e) for e in normalizer.sigma]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> out = normalizer.transform(batch)
        >>> [torch.round(e.mean()).abs() for e in out]
        [tensor(0.), tensor(0.), tensor(0.), tensor(0.)]
        >>> [torch.round(e.std()) for e in out]
        [tensor(1.), tensor(1.), tensor(1.), tensor(1.)]
        >>> x = normalizer.inverse_transform(out)
        >>> [torch.round(e.mean()) for e in x]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> [torch.round(e.std()) for e in x]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        """
        self.batch = [e for e in batch if e is not None]
        self.mu = torch.tensor([e.mean() for e in self.batch])
        self.sigma = torch.tensor([e.std() for e in self.batch])

    def transform(self, x):
        n = len(x)
        out = []
        for i in range(n):
            if self.sigma[i] > 0:
                out.append((x[i] - self.mu[i]) / self.sigma[i])
            else:
                out.append(x[i] - self.mu[i])
        return out

    def inverse_transform(self, x):
        n = len(x)
        out = []
        for i in range(n):
            out.append(x[i] * self.sigma[i] + self.mu[i])
        return out


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    import matplotlib.pyplot as plt
    ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
