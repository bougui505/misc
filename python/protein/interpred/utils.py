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
    cmd.reinitialize()
    pymolname = randomgen.randomstring()
    cmd.load(pdb, pymolname)
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
    seq = cmd.get_fastastr(f'{pymolname} and {selection}')
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
    cartprod = torch.zeros((21 * 2, na, nb))
    for i, aaa in enumerate(onehot_a):
        for j, aab in enumerate(onehot_b):
            cartprod[:, i, j] = torch.cat((aaa, aab))
    return cartprod[None, ...]  # Add the batch dimension


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


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
