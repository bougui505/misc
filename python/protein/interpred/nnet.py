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
from misc.protein.interpred import maps


class InterPred(torch.nn.Module):
    """
    >>> coords_A = maps.get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords_A.shape
    torch.Size([1, 85, 3])
    >>> coords_B = maps.get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA')
    >>> coords_B.shape
    torch.Size([1, 13, 3])

    >>> interpred = InterPred()
    >>> interpred(coords_A, coords_B).shape
    torch.Size([1, 85, 13])
    """
    def __init__(self):
        super(InterPred, self).__init__()
        self.fcn_a = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding='same'))
        self.fcn_b = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding='same'))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, coords_a, coords_b):
        # batchsize, na, spacedim = coords_a.shape
        dmat_a = maps.get_dmat(coords_a)
        dmat_b = maps.get_dmat(coords_b)
        out_a = self.fcn_a(dmat_a)
        out_a = out_a.mean(axis=-1)
        out_b = self.fcn_a(dmat_b)
        out_b = out_b.mean(axis=-1)
        out = torch.einsum('ijk,lmn->ikn', out_a, out_b)
        out = self.sigmoid(out)
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
    from pymol import cmd
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
