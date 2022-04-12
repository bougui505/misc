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

lib1 = cdll.LoadLibrary("lib/initialize_matrix.so")
initialize_matrix_C = lib1.initialize_matrix
lib2 = cdll.LoadLibrary("lib/smith_waterman.so")
traceback_C = lib2.traceback


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
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    cmap_a = cmap_a.astype(float)
    cmap_b = cmap_b.astype(float)
    initialize_matrix_C.restype = ndpointer(dtype=c_double, shape=(na * nb))
    mtx = initialize_matrix_C(c_int(na), c_int(nb), c_void_p(cmap_a.ctypes.data), c_void_p(cmap_b.ctypes.data),
                              c_double(sep_x), c_double(sep_y))
    mtx = mtx.reshape((na, nb))
    return mtx


def traceback(mtx, gap_open=0., gap_extension=0.):
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
    >>> aln = traceback(mtx)
    >>> aln
    {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: -1, 8: -1, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 30, 30: 31, 31: 32, 32: 34, 33: 38, 34: 39, 35: 45, 36: 46, 37: 47, 38: 48, 39: 49, 40: 50, 41: 51, 42: 52, 43: 53, 44: 55, 45: 56, 46: 57, 47: 58, 48: 59, 49: 60, 50: 69, 51: 70, 52: 88, 53: 89, 54: 90, 55: 91, 56: 95, 57: 96, 58: 97, 59: 98, 60: 102, 61: 103, 62: 121, 63: 122, 64: 123, 65: 124, 66: 125, 67: 126, 68: 127, 69: 128, 70: 129, 71: 130, 72: 151, 73: 155, 74: 157, 75: 158, 76: 159, 77: 160, 78: 162, 79: 163, 80: 164, 81: 165, 82: 168, 83: 188, 84: 202, 85: 203, 86: 204, 87: 205}
    """
    na, nb = mtx.shape
    traceback_C.restype = ndpointer(dtype=c_int, shape=na)
    aln = traceback_C(c_int(na), c_int(nb), c_void_p(mtx.ctypes.data), c_double(gap_open), c_double(gap_extension))
    aln = {k: v for k, v in enumerate(aln)}
    return aln


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    from pymol import cmd
    import mapalign
    import matplotlib.pyplot as plt
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
