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

import numpy as np
import scipy.spatial.distance as scidist
from cwrap import initialize_matrix


def get_dmat(coords):
    dmat = scidist.pdist(coords)
    dmat = scidist.squareform(dmat)
    return dmat


def get_cmap(dmat, thr=8.):
    return dmat <= thr


def get_contacts(cmap):
    """
    >>> cmap = np.asarray([[True, False, True], [False, True, False], [True, False, True]])
    >>> contacts = get_contacts(cmap)
    >>> contacts
    (array([0, 0, 1, 2, 2]), array([0, 2, 1, 0, 2]))
    """
    assert cmap.dtype == np.dtype('bool')
    contacts = np.where(cmap)
    return contacts


def sep_weight(sequence_separation):
    weights = np.ones_like(sequence_separation, dtype=float)
    weights[sequence_separation <= 4] = 0.5
    weights[sequence_separation == 5] = 0.75
    return weights


def gaussian(mu, sigma, x):
    g = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return g


def get_seq_sep(contacts):
    s = contacts[0] - contacts[1]
    return s


def traceback(M, i=None, j=None, gap_open=0., gap_extension=0.):
    """
    >>> cmd.reinitialize()
    >>> cmd.load('/home/bougui/pdb/1ycr.cif', 'A_')
    >>> cmd.load('/home/bougui/pdb/1t4e.cif', 'B_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> coords_b = cmd.get_coords('B_ and polymer.protein and chain A and name CA')
    >>> dmat_a = get_dmat(coords_a)
    >>> dmat_b = get_dmat(coords_b)
    >>> cmap_a = get_cmap(dmat_a)
    >>> cmap_b = get_cmap(dmat_b)
    >>> cmap_a.shape, cmap_b.shape
    ((85, 85), (96, 96))
    >>> mtx = initialize_matrix(cmap_a, cmap_b, sep_x=1, sep_y=1)
    >>> aln, score = traceback(mtx, gap_open=2., gap_extension=1.)
    >>> aln
    {83: 78, 82: 77, 81: 76, 80: 76, 79: 76, 78: 76, 77: 76, 76: 76, 75: 76, 74: 76, 73: 76, 72: 76, 71: 76, 70: 76, 69: 75, 68: 74, 67: 73, 66: 72, 65: 71, 64: 71, 63: 71, 62: 70, 61: 69, 60: 68, 59: 67, 58: 66, 57: 65, 56: 64, 55: 63, 54: 62, 53: 61, 52: 60, 51: 59, 50: 58, 49: 57, 48: 56, 47: 55, 46: 54, 45: 53, 44: 52, 43: 51, 42: 50, 41: 49, 40: 48, 39: 47, 38: 45, 37: 35, 36: 34, 35: 33, 34: 32, 33: 31, 32: 30, 31: 29, 30: 29, 29: 29, 28: 29, 27: 29, 26: 29, 25: 29, 24: 29, 23: 28, 22: 28, 21: 28, 20: 28, 19: 27, 18: 26, 17: 25, 16: 24, 15: 23, 14: 21, 13: 20, 12: 18, 11: 17, 10: 16, 9: 15, 8: 14, 7: 13, 6: 12, 5: 12, 4: 12, 3: 12, 2: 11, 1: 10, 0: 9}
    """
    score = 0.
    n, p = M.shape
    alignment = {}
    if i is None or j is None:
        i, j = np.unravel_index(M.argmax(), M.shape)
    alignment[i] = j
    extending = False
    while not (i == 0 or j == 0):
        moves = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        if extending:
            gap_penalty = gap_extension
        else:
            gap_penalty = gap_open
        A = M[moves[0]]  # Align
        D = M[moves[1]] - gap_penalty  # Down
        R = M[moves[2]] - gap_penalty  # Right
        scores = [A, D, R]
        ind = np.argmax(scores)
        if ind != 0:
            extending = True
        else:
            extending = False
        score += scores[ind]
        i, j = moves[ind]
        alignment[i] = j
    return alignment, score


def get_alignment(cmap_a, cmap_b, sep_x, sep_y, gap_open, gap_extension, niter=20):
    """
    >>> cmd.reinitialize()
    >>> cmd.load('data/3u97_A.pdb', 'A_')
    >>> cmd.load('data/2pd0_A.pdb', 'B_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> coords_b = cmd.get_coords('B_ and polymer.protein and chain A and name CA')
    >>> dmat_a = get_dmat(coords_a)
    >>> dmat_b = get_dmat(coords_b)
    >>> cmap_a = get_cmap(dmat_a)
    >>> cmap_b = get_cmap(dmat_b)
    >>> sep_x, sep_y, gap_open, gap_extension = 1, 1, 0, 0
    >>> aln = get_alignment(cmap_a, cmap_b, sep_x, sep_y, gap_open, gap_extension)
    >>> aln
    {82: 201, 81: 201, 80: 201, 79: 201, 78: 201, 77: 201, 76: 194, 75: 194, 74: 194, 73: 194, 72: 194, 71: 194, 70: 194, 69: 194, 68: 194, 67: 194, 66: 194, 65: 194, 64: 194, 63: 194, 62: 194, 61: 194, 60: 194, 59: 194, 58: 194, 57: 194, 56: 194, 55: 194, 54: 194, 53: 194, 52: 194, 51: 194, 50: 194, 49: 194, 48: 194, 47: 194, 46: 194, 45: 194, 44: 194, 43: 194, 42: 194, 41: 194, 40: 194, 39: 194, 38: 194, 37: 194, 36: 194, 35: 194, 34: 194, 33: 194, 32: 194, 31: 194, 30: 194, 29: 194, 28: 194, 27: 194, 26: 194, 25: 194, 24: 194, 23: 194, 22: 194, 21: 194, 20: 194, 19: 194, 18: 194, 17: 194, 16: 194, 15: 194, 14: 194, 13: 190, 12: 190, 11: 190, 10: 190, 9: 190, 8: 190, 7: 190, 6: 189, 5: 189, 4: 189, 3: 189, 2: 189, 1: 117, 0: 116}
    """
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    mtx = initialize_matrix(cmap_a, cmap_b, sep_x, sep_y)
    contacts_a = get_contacts(cmap_a)
    contacts_b = get_contacts(cmap_b)
    sa = get_seq_sep(contacts_a)
    bi = contacts_b[0]
    bj = contacts_b[1]
    for i in range(niter):
        alignment, score = traceback(mtx, gap_open=gap_open, gap_extension=gap_extension)
        logging.info(i)
        bi_ind = [i for i, e in enumerate(bi) if e in alignment]
        bi_aln = bi[np.r_[bi_ind]]
        bj_aln = np.asarray([alignment[e] for e in bj if e in alignment])
        sb = get_seq_sep((bi_aln, bj_aln))
        sa_mesh, sb_mesh = np.meshgrid(sa, sb, indexing='ij')
        mask = ~np.logical_or(np.logical_and(sa_mesh > 0, sb_mesh > 0), np.logical_and(sa_mesh < 0, sb_mesh < 0))
        M_inds = np.meshgrid(contacts_a[1], bj_aln, indexing='ij')
        s_min = np.minimum(np.abs(sa_mesh), np.abs(sb_mesh))
        w = sep_weight(s_min)
        logging.info(w.shape)
        inds = (M_inds[0][~mask], M_inds[1][~mask])
        mtx[inds] = i / (i + 1) * mtx[inds] + w[~mask] / (i + 1)
        logging.info(f'mtx.mean: {mtx.mean()}')
    return alignment


def plot_aln(cmap_a, cmap_b, aln):
    pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    from pymol import cmd
    import matplotlib.pyplot as plt
    # ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
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
