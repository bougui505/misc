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
import copy
import utils


class InterPred(torch.nn.Module):
    """
    >>> coords_A, seq_a = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> coords_A.shape
    torch.Size([1, 85, 3])
    >>> coords_B, seq_b = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA', return_seq=True)
    >>> coords_B.shape
    torch.Size([1, 13, 3])
    >>> interseq = utils.get_inter_seq(seq_a, seq_b)
    >>> interseq.shape
    torch.Size([1, 42, 85, 13])
    >>> cmap_a, cmap_b = interpred.get_input_mats(coords_A, coords_B)
    >>> interpred = InterPred(verbose=True)
    >>> out = interpred(cmap_a, cmap_b, interseq)
    mat_in: torch.Size([1, 44, 224, 224])
    out_fcn: torch.Size([1, 512, 12, 12])
    out_dense: torch.Size([1, 10000])
    out: torch.Size([1, 85, 13])
    """
    def __init__(self, interpolate=True, verbose=False):
        super(InterPred, self).__init__()
        self.interpolate = interpolate
        # AlexNet Convolutions
        layers = [
            torch.nn.Conv2d(in_channels=44, out_channels=96, kernel_size=11, stride=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.fcn = torch.nn.Sequential(*layers)
        # Fully connected of AlexNet
        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=73728, out_features=10000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=10000, out_features=10000),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=10000, out_features=10000),
            torch.nn.Sigmoid()
        ]
        self.fc = torch.nn.Sequential(*layers)
        self.verbose = verbose

    def forward(self, mat_a, mat_b, interseq):
        batch_size, nchannels, na, na = mat_a.shape
        batch_size, nchannels, nb, nb = mat_b.shape
        if self.interpolate:
            mat_a = torch.nn.functional.interpolate(mat_a, size=224)
            mat_b = torch.nn.functional.interpolate(mat_b, size=224)
            interseq = torch.nn.functional.interpolate(interseq, size=224)
        mat_in = torch.cat((mat_a, mat_b, interseq), dim=1)
        if self.verbose:
            print('mat_in:', mat_in.shape)
        out = self.fcn(mat_in)
        if self.verbose:
            print('out_fcn:', out.shape)
        out = self.fc(out)
        if self.verbose:
            print('out_dense:', out.shape)
        out = out.reshape((batch_size, 1, 100, 100))
        if self.interpolate:
            out = torch.nn.functional.interpolate(out, size=(na, nb))
        out = out[:, 0, ...]
        if self.verbose:
            print('out:', out.shape)
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
    import interpred
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
