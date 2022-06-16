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


def latent_sequence(pdb, model, sel='all', latent_dims=512):
    """
    Latent space representation of the protein sequence
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_seq = latent_sequence('1ycr', model, sel='chain A')
    >>> z_seq.shape
    torch.Size([85, 512])
    """
    coords = utils.get_coords(pdb, sel=sel)
    dmat = utils.get_dmat(coords[None, ...])
    with torch.no_grad():
        # FCN model
        z, conv = model(dmat, get_conv=True)
    # log(f'conv: {conv.shape}')  # torch.Size([1, 512, 98, 98])
    z_seq = torch.maximum(conv.max(dim=-1).values, conv.max(dim=-2).values).squeeze().T
    z_seq = z_seq / torch.linalg.norm(z_seq, dim=1)[:, None]
    return z_seq


def get_substitution_matrix(pdb1, pdb2, model, sel1='all', sel2='all', latent_dims=512):
    """
    # >>> import matplotlib.pyplot as plt
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix('1ycr', '7ad0', model, sel1='chain A', sel2='chain D')
    >>> z_sub.shape
    torch.Size([85, 88])

    # >>> _ = plt.matshow(z_sub)
    # >>> _ = plt.colorbar()
    # >>> _ = plt.show()
    """
    z_seq_1 = latent_sequence(pdb1, model, sel=sel1, latent_dims=latent_dims)
    z_seq_2 = latent_sequence(pdb2, model, sel=sel2, latent_dims=latent_dims)
    z_sub = torch.matmul(z_seq_1, z_seq_2.T)
    return z_sub


def get_score_mat(z_sub, gap=-0.1):
    """
    # >>> import matplotlib.pyplot as plt
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix('1ycr', '7c3y', model, sel1='chain A', sel2='chain A')
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


def traceback(M, z_sub, gap=-0.1):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220615_2221.pt')
    Loading FCN model
    >>> z_sub = get_substitution_matrix('1ycr', '7c3y', model, sel1='chain A', sel2='chain A')
    >>> z_sub.shape
    torch.Size([85, 96])
    >>> M = get_score_mat(z_sub)
    >>> M.shape
    (85, 96)
    >>> aln1, aln2 = traceback(M, z_sub)
    >>> aln1
    {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 22, 12: 23, 13: 24, 14: 25, 15: 26, 16: 27, 17: 28, 18: 29, 19: 30, 20: 31, 21: 32, 22: 33, 23: 34, 24: 35, 25: 36, 26: 37, 27: 38, 28: 39, 29: 40, 30: 41, 31: 42, 32: 43, 33: 44, 34: 45, 35: 46, 36: 47, 37: 48, 38: 49, 39: 50, 40: 51, 41: 52, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57, 47: 58, 48: 59, 49: 60, 50: 61, 51: 62, 52: 63, 53: 64, 54: 65, 55: 66, 56: 67, 57: 68, 58: 69, 59: 70, 60: 71, 61: 72, 62: 73, 63: 74, 64: 75, 65: 76, 66: 77, 67: 78, 68: 79, 69: 80, 70: 81, 71: 82, 72: 83, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90, 80: 91, 81: 92, 82: 93, 83: 94}
    >>> aln2
    {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10, 22: 11, 23: 12, 24: 13, 25: 14, 26: 15, 27: 16, 28: 17, 29: 18, 30: 19, 31: 20, 32: 21, 33: 22, 34: 23, 35: 24, 36: 25, 37: 26, 38: 27, 39: 28, 40: 29, 41: 30, 42: 31, 43: 32, 44: 33, 45: 34, 46: 35, 47: 36, 48: 37, 49: 38, 50: 39, 51: 40, 52: 41, 53: 42, 54: 43, 55: 44, 56: 45, 57: 46, 58: 47, 59: 48, 60: 49, 61: 50, 62: 51, 63: 52, 64: 53, 65: 54, 66: 55, 67: 56, 68: 57, 69: 58, 70: 59, 71: 60, 72: 61, 73: 62, 74: 63, 75: 64, 76: 65, 77: 66, 78: 67, 79: 68, 80: 69, 81: 70, 82: 71, 83: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 91: 80, 92: 81, 93: 82, 94: 83}
    """
    n1, n2 = M.shape
    aln1 = dict()
    aln2 = dict()
    i, j = np.unravel_index(M.argmax(), M.shape)
    while i > 0 and j > 0:
        if M[i, j] == M[i - 1, j - 1] + z_sub[i, j]:
            i = i - 1
            j = j - 1
            aln1[i] = j
            aln2[j] = i
        if M[i, j] == M[i - 1, j] + gap:
            i = i - 1
            aln1[i] = '-'
        if M[i, j] == M[i, j - 1] + gap:
            j = j - 1
            aln2[j] = '-'
    aln1 = {k: aln1[k] for k in reversed(aln1)}
    aln2 = {k: aln2[k] for k in reversed(aln2)}
    return aln1, aln2


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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
