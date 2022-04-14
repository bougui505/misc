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
import matplotlib.pyplot as plt
import scipy.spatial.distance as scidist
import tqdm
import cwrap


def get_dmat(coords):
    dmat = scidist.pdist(coords)
    dmat = scidist.squareform(dmat)
    return dmat


def get_cmap(dmat, thr=8.):
    return dmat <= thr


def mapalign(cmap_a,
             cmap_b,
             sep_x_list=[0, 1, 2],
             sep_y_list=[1, 2, 3, 8, 16, 32],
             gap_e_list=[-0.2, -0.1, -0.01, -0.001],
             niter=20,
             progress=True):
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
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))

    # Few minutes to run. Uncomment the following to test it!
    >>> aln, score, sep_x_best, sep_y_best, gap_e_best = mapalign(cmap_a, cmap_b)
    >>> aln
    array([ -1,  -1,   0,   1,   2,   3,  -1,  -1,   4,   5,  -1,  -1,   6,
             7,   8,   9,  10,  11,  12,  13,  14,  19,  20,  21,  22,  23,
            31,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
            48,  49,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
            64,  65,  66,  67,  68,  69, 116, 117, 118, 119, 120, 121, 122,
           123, 124, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158,
           159, 160, 161, 162, 163, 164, 165,  -1,  -1,  -1])
    >>> aln.shape
    (88,)
    >>> score
    98.91796030178082
    >>> sep_x_best, sep_y_best, gap_e_best
    (2, 16, -0.001)
    """
    aln, score, sep_x_best, sep_y_best, gap_e_best = cwrap.get_alignment(cmap_a,
                                                                         cmap_b,
                                                                         sep_x_list=sep_x_list,
                                                                         sep_y_list=sep_y_list,
                                                                         gap_extension_list=gap_e_list,
                                                                         niter=niter,
                                                                         progress=progress)
    return aln, score, sep_x_best, sep_y_best, gap_e_best


def get_aln_b(aln_a, nb):
    """
    >>> aln_a = np.asarray([ -1,  -1,   0,   1,   2,   3,  -1,  -1,   4,   5,  -1,  -1,   6, 7,   8,   9,  10,  11,  12,  13,  14,  19,  20,  21,  22,  23, 31,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, 48,  49,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, 64,  65,  66,  67,  68,  69, 116, 117, 118, 119, 120, 121, 122, 123, 124, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,  -1,  -1,  -1])
    >>> aln_a.shape
    (88,)
    >>> aln_b = get_aln_b(aln_a, 215)
    >>> aln_b
    array([ 2.,  3.,  4.,  5.,  8.,  9., 12., 13., 14., 15., 16., 17., 18.,
           19., 20., -1., -1., -1., -1., 21., 22., 23., 24., 25., -1., -1.,
           -1., -1., -1., -1., -1., 26., -1., -1., -1., -1., 27., 28., 29.,
           30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., -1., -1.,
           -1., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52.,
           53., 54., 55., 56., 57., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 58.,
           59., 60., 61., 62., 63., 64., 65., 66., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., 67., 68., 69., 70., 71., 72., -1., -1., -1., 73., 74.,
           75., 76., 77., 78., 79., 80., 81., 82., 83., 84., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
           -1., -1., -1., -1., -1., -1., -1.])
    """
    aln_b = -np.ones(nb)
    ai_aln = np.where(aln_a != -1)[0]
    bi_aln = aln_a[ai_aln]
    aln_b[bi_aln] = ai_aln
    return aln_b


def get_aligned_maps(cmap_a, cmap_b, aln, full=False):
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
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))
    >>> aln = np.asarray([ -1,  -1,   0,   1,   2,   3,  -1,  -1,   4,   5,  -1,  -1,   6, 7,   8,   9,  10,  11,  12,  13,  14,  19,  20,  21,  22,  23, 31,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, 48,  49,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, 64,  65,  66,  67,  68,  69, 116, 117, 118, 119, 120, 121, 122, 123, 124, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,  -1,  -1,  -1])
    >>> aln.shape
    (88,)

    Returns the maps aligned in the frame of cmap_a
    >>> cmap_a_aln, cmap_b_aln = get_aligned_maps(cmap_a, cmap_b, aln)
    >>> cmap_a_aln.shape
    (79, 79)
    >>> cmap_a_aln.shape
    (79, 79)

    Returns the maps aligned in the frame of cmap_b
    >>> cmap_a_aln, cmap_b_aln = get_aligned_maps(cmap_a, cmap_b, aln, full=True)
    >>> cmap_a_aln.shape
    (215, 215)
    >>> cmap_b_aln.shape
    (215, 215)
    """
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    ai_aln = np.where(aln != -1)[0]
    bi_aln = aln[ai_aln]
    if not full:  # Only get the aligned parts
        cmap_a_aln = cmap_a[ai_aln, :][:, ai_aln]
        cmap_b_aln = cmap_b[bi_aln, :][:, bi_aln]
    else:  # get the FULL matrices with zeros in insertion regions
        if na < nb:
            cmap_a_aln = np.zeros_like(cmap_b)
            cmap_a_aln[:na, :na] = cmap_a
            cmap_a_aln[bi_aln, :] = cmap_a_aln[ai_aln, :]
            cmap_a_aln[:, bi_aln] = cmap_a_aln[:, ai_aln]
            cmap_b_aln = cmap_b
        else:
            cmap_a_aln = cmap_a
            cmap_b_aln = np.zeros_like(cmap_a)
            cmap_b_aln[:nb, :nb] = cmap_b
            cmap_b_aln[ai_aln, :] = cmap_b_aln[bi_aln, :]
            cmap_b_aln[:, ai_aln] = cmap_b_aln[:, bi_aln]
    return cmap_a_aln, cmap_b_aln


def get_score(cmap_a, cmap_b, aln):
    """
    The score is the number of contacts common in the two maps aligned over the total number of contacts for cmap_a
    >>> cmd.reinitialize()
    >>> cmd.load('data/3u97_A.pdb', 'A_')
    >>> cmd.load('data/2pd0_A.pdb', 'B_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> coords_b = cmd.get_coords('B_ and polymer.protein and chain A and name CA')
    >>> dmat_a = get_dmat(coords_a)
    >>> dmat_b = get_dmat(coords_b)
    >>> cmap_a = get_cmap(dmat_a)
    >>> cmap_b = get_cmap(dmat_b)
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))
    >>> aln = np.asarray([ -1,  -1,   0,   1,   2,   3,  -1,  -1,   4,   5,  -1,  -1,   6, 7,   8,   9,  10,  11,  12,  13,  14,  19,  20,  21,  22,  23, 31,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, 48,  49,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, 64,  65,  66,  67,  68,  69, 116, 117, 118, 119, 120, 121, 122, 123, 124, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,  -1,  -1,  -1])
    >>> aln.shape
    (88,)
    >>> score = get_score(cmap_a, cmap_b, aln)
    >>> score
    0.6120162932790224
    """
    cmap_a_aln, cmap_b_aln = get_aligned_maps(cmap_a, cmap_b, aln, full=False)
    comm = np.logical_and(cmap_a_aln, cmap_b_aln)
    score = comm.sum() / cmap_a.sum()  # min(cmap_a.sum(), cmap_b.sum())
    return score


def plot_aln(cmap_a, cmap_b, aln, full=False, outfilename=None):
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
    >>> cmap_a.shape, cmap_b.shape
    ((88, 88), (215, 215))
    >>> aln = np.asarray([ -1,  -1,   0,   1,   2,   3,  -1,  -1,   4,   5,  -1,  -1,   6, 7,   8,   9,  10,  11,  12,  13,  14,  19,  20,  21,  22,  23, 31,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, 48,  49,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, 64,  65,  66,  67,  68,  69, 116, 117, 118, 119, 120, 121, 122, 123, 124, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,  -1,  -1,  -1])
    >>> aln.shape
    (88,)
    >>> plot_aln(cmap_a, cmap_b, aln)

    >>> plot_aln(cmap_a, cmap_b, aln, full=True)
    """
    cmap_a_aln, cmap_b_aln = get_aligned_maps(cmap_a, cmap_b, aln, full=full)
    ai, aj = np.where(cmap_a_aln > 0)
    bi, bj = np.where(cmap_b_aln > 0)
    plt.scatter(bi, bj, s=16., c='gray', alpha=.5, label='cmap_b')
    plt.scatter(ai, aj, s=1., c='blue', label='cmap_a')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    if outfilename is not None:
        plt.savefig(outfilename)
    else:
        plt.show()


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
    parser.add_argument('-p1', '--pdb1')
    parser.add_argument('-p2', '--pdb2')
    parser.add_argument('-s1', '--sel1', required=False, default='all')
    parser.add_argument('-s2', '--sel2', required=False, default='all')
    parser.add_argument(
        '--sep_x',
        type=int,
        default=2,
        help=
        'Parameter to compute the STD of the gaussian: s_std=sep_y*(1+(s_min-2)**sep_x), with s_min the min sequence separation for cmap_a and cmap_b of the considered contacts. (default=2)'
    )
    parser.add_argument(
        '--sep_y',
        type=int,
        default=16,
        help=
        'Parameter to compute the STD of the gaussian: s_std=sep_y*(1+(s_min-2)**sep_x), with s_min the min sequence separation for cmap_a and cmap_b of the considered contacts. (default=16)'
    )
    parser.add_argument('--gap_e',
                        type=float,
                        default=-0.001,
                        help='Gap extension penalty. MUST BE negative (default=-0.001).')
    parser.add_argument('--show', action='store_true', help='Show the contact map alignment')
    parser.add_argument('--save', help='Save the contact map alignment in the given filename')
    parser.add_argument('--full',
                        action='store_true',
                        help='Display the full contact map alignemnt. Not only the aligned contacts')
    parser.add_argument('--hpo', help='Hyperparameter optimization for sep_x, sep_y and gap_e', action='store_true')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    log(args.pdb1)
    log(args.pdb2)
    cmd.load(args.pdb1, 'A_')
    cmd.load(args.pdb2, 'B_')
    coords_a = cmd.get_coords(f'A_ and polymer.protein and name CA and {args.sel1}')
    coords_b = cmd.get_coords(f'B_ and polymer.protein and name CA and {args.sel2}')
    dmat_a = get_dmat(coords_a)
    dmat_b = get_dmat(coords_b)
    cmap_a = get_cmap(dmat_a)
    cmap_b = get_cmap(dmat_b)
    log(f'cmap_a.shape: {cmap_a.shape}')
    log(f'cmap_b.shape: {cmap_b.shape}')
    if args.hpo:
        sep_x_list = [0, 1, 2]
        sep_y_list = [1, 2, 3, 8, 16, 32]
        gap_e_list = [-0.2, -0.1, -0.01, -0.001]
    else:
        sep_x_list = [args.sep_x]
        sep_y_list = [args.sep_y]
        gap_e_list = [args.gap_e]
    aln, score, sep_x_best, sep_y_best, gap_e_best = mapalign(cmap_a,
                                                              cmap_b,
                                                              sep_x_list=sep_x_list,
                                                              sep_y_list=sep_y_list,
                                                              gap_e_list=gap_e_list,
                                                              progress=args.hpo)
    log(f'score: {score:.4f}')
    print(f'score: {score:.4f}')
    native_contacts_score = get_score(cmap_a, cmap_b, aln)
    log(f'native_contacts_score: {native_contacts_score:.4f}')
    print(f'native_contacts_score: {native_contacts_score:.4f}')
    if args.show or args.save is not None:
        plot_aln(cmap_a, cmap_b, aln, full=args.full, outfilename=args.save)
    # >>> sep_x_best, sep_y_best, gap_e_best
    # (2, 16, -0.001)
