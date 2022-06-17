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
from misc.protein.sscl import utils
from misc.protein.sscl import encoder
import matplotlib.pyplot as plt


class MapAlign():
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> ma = MapAlign(model=model, pdb1='1ycr', pdb2='7ad0', sel1='chain A', sel2='chain D')
    >>> ma.cmap1.shape
    (85, 85)
    >>> ma.cmap2.shape
    (88, 88)
    >>> ma.cmap1_aln.shape
    (88, 88)
    >>> ma.cmap2_aln.shape
    (88, 88)
    """
    def __init__(self, model, pdb1=None, pdb2=None, dmat1=None, dmat2=None, sel1='all', sel2='all', gap=0.):
        """
        """
        coords1 = utils.get_coords(pdb1, sel=sel1)
        coords2 = utils.get_coords(pdb2, sel=sel2)
        self.dmat1 = utils.get_dmat(coords1[None, ...])
        self.dmat2 = utils.get_dmat(coords2[None, ...])
        self.zsub = get_substitution_matrix(model=model, dmat1=self.dmat1, dmat2=self.dmat2)
        self.M = get_score_mat(self.zsub)
        self.gap = gap
        self.aln1, self.aln2 = traceback(self.M, self.zsub, gap=self.gap)
        self.cmap1 = utils.get_cmap(self.dmat1).squeeze().numpy()
        self.cmap2 = utils.get_cmap(self.dmat2).squeeze().numpy()
        self.aln = np.asarray(list(self.aln1.values()))
        self.cmap1_aln, self.cmap2_aln = get_aligned_maps(self.cmap1, self.cmap2, self.aln, full=True)

    def plot(self, full=True, outfilename=None):
        plot_aln(self.cmap1, self.cmap2, self.aln, full=full, outfilename=outfilename)


def latent_sequence(model, pdb=None, dmat=None, sel='all', latent_dims=512):
    """
    Latent space representation of the protein sequence
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_seq = latent_sequence(pdb='1ycr', model=model, sel='chain A')
    >>> z_seq.shape
    torch.Size([85, 512])
    """
    if pdb is not None:
        coords = utils.get_coords(pdb, sel=sel)
        dmat = utils.get_dmat(coords[None, ...])
    with torch.no_grad():
        # FCN model
        z, conv = model(dmat, get_conv=True)
    # log(f'conv: {conv.shape}')  # torch.Size([1, 512, 98, 98])
    z_seq = torch.maximum(conv.max(dim=-1).values, conv.max(dim=-2).values).squeeze().T
    z_seq = z_seq / torch.linalg.norm(z_seq, dim=1)[:, None]
    return z_seq


def get_substitution_matrix(model,
                            pdb1=None,
                            pdb2=None,
                            dmat1=None,
                            dmat2=None,
                            sel1='all',
                            sel2='all',
                            latent_dims=512):
    """
    # >>> import matplotlib.pyplot as plt
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix(pdb1='1ycr', pdb2='7ad0', model=model, sel1='chain A', sel2='chain D')
    >>> z_sub.shape
    torch.Size([85, 88])

    # >>> _ = plt.matshow(z_sub)
    # >>> _ = plt.colorbar()
    # >>> _ = plt.show()
    """
    z_seq_1 = latent_sequence(pdb=pdb1, dmat=dmat1, model=model, sel=sel1, latent_dims=latent_dims)
    z_seq_2 = latent_sequence(pdb=pdb2, dmat=dmat2, model=model, sel=sel2, latent_dims=latent_dims)
    z_sub = torch.matmul(z_seq_1, z_seq_2.T)
    return z_sub


def get_score_mat(z_sub, gap=0.):
    """
    # >>> import matplotlib.pyplot as plt
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix(pdb1='1ycr', pdb2='7c3y', model=model, sel1='chain A', sel2='chain A')
    >>> z_sub.shape
    torch.Size([85, 96])
    >>> M = get_score_mat(z_sub)
    >>> M.shape
    (85, 96)

    # >>> _ = plt.matshow(M)
    # >>> _ = plt.colorbar()
    # >>> _ = plt.show()
    """
    n1, n2 = z_sub.shape
    M = np.zeros((n1, n2))
    for i in range(1, n1):
        for j in range(1, n2):
            M[i, j] = max(0, M[i - 1, j - 1] + z_sub[i, j], M[i - 1, j] + gap, M[i, j - 1] + gap)
    return M


def traceback(M, z_sub, gap=0.):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix(pdb1='1ycr', pdb2='7c3y', model=model, sel1='chain A', sel2='chain A')
    >>> z_sub.shape
    torch.Size([85, 96])
    >>> M = get_score_mat(z_sub)
    >>> M.shape
    (85, 96)
    >>> aln1, aln2 = traceback(M, z_sub)
    >>> aln1
    {0: 0, 1: 1, 2: 2, 3: 7, 4: 15, 5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 22, 12: 23, 13: 24, 14: 25, 15: 26, 16: 27, 17: 28, 18: 29, 19: 30, 20: 31, 21: 32, 22: 33, 23: 34, 24: 35, 25: 36, 26: 37, 27: 38, 28: 39, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49, 39: 50, 40: 51, 41: 52, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57, 47: 58, 48: 59, 49: 60, 50: 61, 51: 62, 52: 63, 53: 64, 54: 65, 55: 66, 56: 67, 57: 68, 58: 69, 59: 70, 60: 71, 61: 72, 62: 73, 63: 74, 64: 75, 65: 76, 66: 77, 67: 78, 68: 79, 69: 80, 70: 81, 71: 82, 72: 83, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90, 80: 91, 81: 92, 82: 93, 83: 94, 84: 95}
    >>> aln2
    {0: 0, 1: 1, 2: 2, 3: None, 4: None, 5: None, 6: None, 7: 3, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10, 22: 11, 23: 12, 24: 13, 25: 14, 26: 15, 27: 16, 28: 17, 29: 18, 30: 19, 31: 20, 32: 21, 33: 22, 34: 23, 35: 24, 36: 25, 37: 26, 38: 27, 39: 28, 40: 29, 41: 30, 42: 31, 43: 32, 44: 33, 45: 34, 46: 35, 47: 36, 48: 37, 49: 38, 50: 39, 51: 40, 52: 41, 53: 42, 54: 43, 55: 44, 56: 45, 57: 46, 58: 47, 59: 48, 60: 49, 61: 50, 62: 51, 63: 52, 64: 53, 65: 54, 66: 55, 67: 56, 68: 57, 69: 58, 70: 59, 71: 60, 72: 61, 73: 62, 74: 63, 75: 64, 76: 65, 77: 66, 78: 67, 79: 68, 80: 69, 81: 70, 82: 71, 83: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 91: 80, 92: 81, 93: 82, 94: 83, 95: 84}
    """
    n1, n2 = M.shape
    aln1 = dict()
    aln2 = dict()
    i, j = np.unravel_index(M.argmax(), M.shape)
    aln1[i] = j
    aln2[j] = i
    while i > 0 and j > 0:
        if M[i, j] == M[i - 1, j - 1] + z_sub[i, j]:
            i = i - 1
            j = j - 1
            aln1[i] = j
            aln2[j] = i
            continue
        if M[i, j] == M[i - 1, j] + gap:
            i = i - 1
            aln1[i] = None
            continue
        if M[i, j] == M[i, j - 1] + gap:
            j = j - 1
            aln2[j] = None
            continue
    aln1 = {k: aln1[k] for k in reversed(aln1)}
    aln2 = {k: aln2[k] for k in reversed(aln2)}
    return aln1, aln2


def get_aligned_maps(cmap_a, cmap_b, aln, full=False):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> ma = MapAlign(model=model, pdb1='1ycr', pdb2='7ad0', sel1='chain A', sel2='chain D')
    >>> ma.cmap1.shape
    (85, 85)
    >>> ma.cmap2.shape
    (88, 88)
    >>> aln = np.asarray(list(ma.aln1.values()))
    >>> aln
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 86, 87])
    >>> aln.shape
    (85,)
    >>> cmap1_aln, cmap2_aln = get_aligned_maps(ma.cmap1, ma.cmap2, aln, full=True)
    >>> cmap1_aln.shape
    (88, 88)
    >>> cmap2_aln.shape
    (88, 88)

    """
    na, na = cmap_a.shape
    nb, nb = cmap_b.shape
    ai_aln = np.where(aln != None)[0]
    bi_aln = aln[ai_aln]
    if not full:  # Only get the aligned parts
        cmap_a_aln = cmap_a[ai_aln, :][:, ai_aln]
        cmap_b_aln = cmap_b[bi_aln, :][:, bi_aln]
    else:  # get the FULL matrices with zeros in insertion regions
        if na <= nb:
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


def plot_aln(cmap_a, cmap_b, aln, full=False, outfilename=None):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> ma = MapAlign(model=model, pdb1='1ycr', pdb2='7ad0', sel1='chain A', sel2='chain D')
    >>> plot_aln(ma.cmap1, ma.cmap2, ma.aln)
    """
    cmap_a_aln, cmap_b_aln = get_aligned_maps(cmap_a, cmap_b, aln, full=full)
    ai, aj = np.where(cmap_a_aln > 0.5)
    bi, bj = np.where(cmap_b_aln > 0.5)
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
    parser.add_argument('--pdb1')
    parser.add_argument('--pdb2')
    parser.add_argument('--sel1')
    parser.add_argument('--sel2')
    parser.add_argument('--model', help='Model to load', metavar='model.pt', default='models/sscl_fcn_20220615_2221.pt')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    model = encoder.load_model(args.model)
    ma = MapAlign(model=model, pdb1=args.pdb1, pdb2=args.pdb2, sel1=args.sel1, sel2=args.sel2)
    ma.plot(full=False)
