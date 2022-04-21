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
import tqdm
from misc.mapalign import cwrap
from misc.mapalign import mapalign


def get_wv(M, t):
    w, v = np.linalg.eigh(M)
    idx = np.abs(w).argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    w = w[:t]
    v = v[:, :t]
    return np.sqrt(np.abs(w)) * v


def initialize_eigen(cmap_a, cmap_b, t=None):
    """
    See: https://doi.org/10.1093/bioinformatics/btq402

    >>> cmd.reinitialize()
    >>> cmd.load('data/3u97_A.pdb', 'A_')
    >>> cmd.load('data/2pd0_A.pdb', 'B_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> coords_b = cmd.get_coords('B_ and polymer.protein and chain A and name CA')
    >>> dmat_a = mapalign.get_dmat(coords_a)
    >>> dmat_b = mapalign.get_dmat(coords_b)
    >>> cmap_a = mapalign.get_cmap(dmat_a)
    >>> cmap_b = mapalign.get_cmap(dmat_b)
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))
    >>> S = initialize_eigen(cmap_a, cmap_b)
    >>> S.shape
    (88, 215)

    # >>> plt.matshow(S)
    # >>> plt.colorbar()
    # >>> plt.show()
    """
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    if t is None:
        t = min(na, nb)
    v_a = get_wv(cmap_a, t=t)
    v_b = get_wv(cmap_b, t=t)
    S = v_a.dot(v_b.T)
    return S


def get_alignment(cmap_a,
                  cmap_b,
                  gap_open=-1.,
                  gap_extension_list=[-0.2, -0.1, -0.01, -0.001],
                  niter=20,
                  progress=False,
                  return_mtx=False):
    """
    >>> cmd.reinitialize()
    >>> cmd.load('data/3u97_A.pdb', 'A_')
    >>> cmd.load('data/2pd0_A.pdb', 'B_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> coords_b = cmd.get_coords('B_ and polymer.protein and chain A and name CA')
    >>> dmat_a = mapalign.get_dmat(coords_a)
    >>> dmat_b = mapalign.get_dmat(coords_b)
    >>> cmap_a = mapalign.get_cmap(dmat_a)
    >>> cmap_b = mapalign.get_cmap(dmat_b)
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))

    >>> aln, score, gap_e = get_alignment(cmap_a, cmap_b, gap_extension_list=[-0.01, -0.001], progress=True, niter=20)
    >>> aln
    array([ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
            -1,  -1,  -1,  -1,  -1,  54,  55,  56,  57,  58,  59,  60,  61,
            62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
            75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  -1, 103,
           104, 105, 106, 107, 108, 116, 117, 118, 119, 120, 121, 122, 123,
           124, 125, 126, 127, 128, 129, 130,  -1,  -1, 151, 152, 153, 154,
           155, 156, 157, 158, 159, 160, 161, 162, 163, 164], dtype=int32)
    >>> score
    14.288268879452984
    """
    cmap_a = cmap_a.astype(float)
    cmap_b = cmap_b.astype(float)
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    score_max = 0.
    if progress:
        total = len(gap_extension_list) * niter
        pbar = tqdm.tqdm(total=total)
    mtx_ini = initialize_eigen(cmap_a, cmap_b, t=min(na, nb))
    for gap_e in gap_extension_list:
        mtx = mtx_ini.copy()
        aln, score = cwrap.traceback(mtx, gap_open=gap_open, gap_extension=gap_e)
        for i in range(niter):
            cmap_a_aln, cmap_b_aln = mapalign.get_aligned_maps(cmap_a, cmap_b, aln, full=False)
            ai_aln = np.where(aln != -1)[0]
            bi_aln = aln[ai_aln]
            mtx[ai_aln][:, bi_aln] = initialize_eigen(cmap_a_aln, cmap_b_aln, t=cmap_a_aln.shape[0])
            aln, score = cwrap.traceback(mtx, gap_open=gap_open, gap_extension=gap_e)
            if score >= score_max:
                score_max = score
                aln_best = aln
                gap_e_best = gap_e
            if progress:
                pbar.set_description(f'score: {score_max:.3f}')
                pbar.update(1)
    if progress:
        pbar.close()
    if return_mtx:
        return aln_best, score_max, gap_e_best, mtx
    else:
        return aln_best, score_max, gap_e_best


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    from pymol import cmd
    import matplotlib.pyplot as plt
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
