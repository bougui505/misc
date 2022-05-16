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
import torch.nn as nn
import utils


def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))


class InterPred(nn.Module):
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
    >>> interpred = InterPred()
    >>> out = interpred(cmap_a, cmap_b, interseq)
    >>> out.shape
    torch.Size([1, 85, 13])
    """
    def __init__(self, n_class=1, interpolate=True):
        super().__init__()
        self.interpolate = interpolate
        self.dconv_down1 = double_conv(44, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, mat_a, mat_b, interseq):
        batch_size, nchannels, na, na = mat_a.shape
        batch_size, nchannels, nb, nb = mat_b.shape
        if self.interpolate:
            mat_a = torch.nn.functional.interpolate(mat_a, size=224)
            mat_b = torch.nn.functional.interpolate(mat_b, size=224)
            interseq = torch.nn.functional.interpolate(interseq, size=224)
        x = torch.cat((mat_a, mat_b, interseq), dim=1)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = torch.sigmoid(out)

        if self.interpolate:
            out = torch.nn.functional.interpolate(out, size=(na, nb))
        out = out[:, 0, ...]

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
