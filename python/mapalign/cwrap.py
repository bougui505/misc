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

from ctypes import cdll, c_double, c_int, c_void_p
from numpy.ctypeslib import ndpointer
import numpy as np
import tqdm


def initialize_matrix(cmap_a, cmap_b, sep_x, sep_y):
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
    >>> mtx = initialize_matrix(cmap_a, cmap_b, sep_x=2, sep_y=1)
    >>> mtx.sum()
    69962.87346834535
    >>> mtx.shape
    (88, 215)

    # >>> _ = plt.matshow(mtx)
    # >>> _ = plt.colorbar()
    # >>> _ = plt.show()

    """
    lib = cdll.LoadLibrary("lib/initialize_matrix.so")
    initialize_matrix_C = lib.initialize_matrix
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    cmap_a = cmap_a.astype(float)
    cmap_b = cmap_b.astype(float)
    initialize_matrix_C.restype = ndpointer(dtype=c_double, shape=(na * nb))
    mtx = initialize_matrix_C(c_int(na), c_int(nb), c_void_p(cmap_a.ctypes.data), c_void_p(cmap_b.ctypes.data),
                              c_double(sep_x), c_double(sep_y))
    mtx = mtx.reshape((na, nb))
    return mtx


def traceback(mtx, gap_open=-1., gap_extension=-0.1):
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
    >>> mtx = initialize_matrix(cmap_a, cmap_b, sep_x=2, sep_y=1)
    >>> aln, score = traceback(mtx)
    >>> aln
    array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
            27,  28,  29,  30,  31,  32,  39,  44,  45,  46,  47,  48,  49,
            50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  69,  70,
            88,  89,  90,  98,  99, 100, 101, 102, 103, 104, 121, 122, 123,
           124, 125, 126, 127, 128, 129, 130, 151, 155, 156, 157, 158, 159,
           160, 161, 162, 163, 164, 165, 168, 169, 170, 171])
    >>> score
    509.77545076565457
    """
    lib = cdll.LoadLibrary("lib/smith_waterman.so")
    traceback_C = lib.traceback
    na, nb = mtx.shape
    # Return shape is na+1 as we add the score as the last element of the returned value
    traceback_C.restype = ndpointer(dtype=c_double, shape=na + 1)
    aln = traceback_C(c_int(na), c_int(nb), c_void_p(mtx.ctypes.data), c_double(gap_open), c_double(gap_extension))
    score = aln[-1]
    aln = np.int_(aln[:-1])
    # aln = {k: v for k, v in enumerate(aln)}
    return aln, score


def get_alignment(cmap_a,
                  cmap_b,
                  sep_x_list=[0, 1, 2],
                  sep_y_list=[1, 2, 4, 8, 16, 32],
                  gap_open=-1.,
                  gap_extension_list=[-0.2, -0.1, -0.01, -0.001],
                  niter=20,
                  progress=False):
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
    >>> aln, score = get_alignment(cmap_a, cmap_b, sep_x_list=[2], sep_y_list=[1], gap_extension_list=[-0.1])
    >>> aln
    array([ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
            -1,  -1,  33,  34,  35,  -1,  -1,  -1,  -1,  36,  37,  -1,  -1,
            -1,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  55,  56,
            57,  58,  59,  60,  61,  81,  82,  83,  84,  85,  86,  87,  88,
            89,  90,  91,  92,  93,  94,  96,  97,  98,  99, 100, 101, 102,
           119, 120, 121, 122, 123, 124, 143, 144, 145, 146, 147, 150, 151,
           152, 153, 154,  -1, 155, 156, 157, 158, 159, 160])
    >>> score
    72.88997596280687
    """
    cmap_a = cmap_a.astype(float)
    cmap_b = cmap_b.astype(float)
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    lib = cdll.LoadLibrary("lib/smith_waterman.so")
    update_mtx_C = lib.update_mtx
    update_mtx_C.restype = ndpointer(dtype=c_double, shape=na * nb)
    score_max = 0.
    if progress:
        total = len(sep_x_list) * len(sep_y_list) * len(gap_extension_list) * niter
        pbar = tqdm.tqdm(total=total)
    for sep_x in sep_x_list:
        for sep_y in sep_y_list:
            mtx_ini = initialize_matrix(cmap_a, cmap_b, sep_x, sep_y)
            for gap_e in gap_extension_list:
                mtx = mtx_ini.copy()
                aln, score = traceback(mtx, gap_open=gap_open, gap_extension=gap_e)
                for i in range(niter):
                    mtx = update_mtx_C(c_int(na), c_int(nb), c_void_p(aln.ctypes.data), c_void_p(mtx.ctypes.data),
                                       c_void_p(cmap_a.ctypes.data), c_void_p(cmap_b.ctypes.data), i)
                    mtx = mtx.reshape((na, nb))
                    aln, score = traceback(mtx, gap_open=gap_open, gap_extension=gap_e)
                    if score >= score_max:
                        score_max = score
                        aln_best = aln
                    log(f'iteration: {i}')
                    log(f'score: {score:.3f}')
                    if progress:
                        pbar.set_description(f'score: {score_max:.3f}')
                        pbar.update(1)
    log(f'score_max: {score_max:.3f}')
    if progress:
        pbar.close()
    return aln_best, score_max


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    from pymol import cmd
    import mapalign
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
