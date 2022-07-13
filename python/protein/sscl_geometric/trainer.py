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
import os
import BLASTloader
import torch
import encoder


def collate_fn(batch):
    return batch


def get_batch_test():
    """
    >>> batch = get_batch_test()
    >>> len(batch)
    3
    >>> batch
    [(Data(), Data()), (Data(), Data()), (Data(edge_index=[2, 717], node_id=[154], num_nodes=154, x=[154, 20]), Data(edge_index=[2, ...], node_id=[...], num_nodes=..., x=[..., 20]))]
    """
    dataset = BLASTloader.PDBdataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4, collate_fn=collate_fn)
    for batch in dataloader:
        break
    return batch


def forward_batch(batch, model):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.GCN()
    >>> out = forward_batch(batch, model)
    >>> [(z_anchor.shape, z_positive.shape) for z_anchor, z_positive in out]
    [(torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    out = []
    for anchor, positive in batch:
        if anchor.x is not None and positive.x is not None:
            z_anchor = model(anchor)
            z_positive = model(positive)
            out.append((z_anchor, z_positive))
    return out


def get_contrastive_loss(out, tau=1.):
    """
    >>> n = 3
    >>> out = [(torch.randn(1, 512), torch.randn(1, 512)) for i in range(n)]
    >>> loss = get_contrastive_loss(out)
    >>> loss
    tensor(...)
    """
    n = len(out)
    z_anchor_list = [e[0] for e in out]
    z_positive_list = [e[1] for e in out]
    loss = 0.
    for i in range(n):
        z_full_i = z_anchor_list[i]
        # z_fragment_i = z_fragment_list[i]
        den = 0.
        for j in range(n):
            z_full_j = z_anchor_list[j]
            z_fragment_j = z_positive_list[j]
            if i == j:
                sim_num = torch.matmul(z_full_i, z_fragment_j.T)
                # log(f'z_full_i: {z_full_i}')
                # log(f'sim_num: {sim_num}')
                num = torch.exp(sim_num / tau)
            else:
                sim_den = torch.matmul(z_full_i, z_full_j.T)
                # log(f'sim_den: {sim_den}')
                den += torch.exp(sim_den / tau)
        # log(f'num:{num}, den: {den}')
        loss -= torch.log((num + 1e-8) / (den + 1e-8))
    if n > 0:
        loss = loss / n
    loss = torch.squeeze(loss)
    assert not torch.isnan(loss), 'ERROR: loss is nan'
    return loss


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
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
