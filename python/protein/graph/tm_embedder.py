#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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

import proteingraph
import messagepassing
from torch_geometric.loader import DataLoader
import torch
from misc.protein.pocket_align.pocket_sim import pairwise_pocket_sim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_off_diag(a):
    """
    see: https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379/2
    """
    n = a.shape[0]
    return a.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)


def tmloss(out, tmscore):
    tm_pred = out.matmul(out.T)
    return ((tm_pred - tmscore) ** 2).mean(), tm_pred


def learn(pocketfile, radius=6.0, batch_size=8, n_epochs=100, device=None):
    if device is None:
        device = DEVICE
    print(f"Training on {device}")
    dataset = proteingraph.Dataset(
        txtfile=pocketfile, radius=radius, return_pyg_graph=True
    )
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gcn = messagepassing.GCN(n_n=58, n_e=1, n_o=256, embedding_dim=512, normalize=True)
    gcn = gcn.to(device)
    optimizer = torch.optim.Adam(gcn.parameters())
    for epoch in range(n_epochs):
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            pocketlist = [e[1:] for e in batch.y]
            tmscore, _ = pairwise_pocket_sim(pocketlist=pocketlist, radius=radius)
            tmscore = torch.from_numpy(tmscore).to(device)
            batch = batch.to(device)
            out = gcn(
                batch.x, batch.edge_index, batch.edge_attr, batch_index=batch.batch
            )
            lossval, tm_pred = tmloss(out, tmscore)
            lossval.backward()
            optimizer.step()
            print(
                f"epoch: {epoch}|step: {i}|loss: {lossval:.4g}|tm_pred: {tm_pred.mean():.4g}|tm_score {tmscore.mean()}"
            )


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--pocketfile")
    parser.add_argument("--device")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()
    if args.pocketfile is not None:
        learn(args.pocketfile, device=args.device)
