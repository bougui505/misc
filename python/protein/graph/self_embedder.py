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
from torch.nn.functional import normalize

from torch.optim import optimizer
import numpy as np

import proteingraph
import messagepassing
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
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


def split_feaures(batch):
    res_batch = batch[:, :21]
    atom_batch = batch[:, 21:]
    return res_batch, atom_batch


def success_rate(pred, target):
    """
    pred.shape: torch.Size([BS, 21]) for residue and torch.Size([BS, 37]) for atoms
    """
    _, ind_pred = pred.max(dim=1)
    _, ind_target = target.max(dim=1)
    bs = ind_target.shape[0]
    rate = (ind_pred == ind_target).sum() / bs
    return rate


def maskingLoss(out, batch):
    """
    Assuming batch size (BS) is 16
    out.shape: [torch.Size([284, 58]), torch.Size([320, 58]), torch.Size([316, 58]), ...
    DataBatch(x=[5188, 58], edge_index=[2, 65462], edge_attr=[65462, 1], y=[16], masked_features=[928], masked_atom_id=[16], batch=[5188], ptr=[17])
    """
    pred_mask = []
    for i, e in enumerate(out):
        masked_atom_id = batch.masked_atom_id[i]
        pred_mask.append(e[masked_atom_id])
    pred_mask = torch.cat(pred_mask)  # torch.Size([928])
    pred_mask = pred_mask.reshape(-1, 58)  # torch.Size([16, 58])
    target = batch.masked_features.reshape(-1, 58)  # torch.Size([16, 58])
    # The sum must be 2 (1 hot for the residue type for the 21 dimension and 1 hot for the atom type for the 37 last dimension):
    assert (target.sum(dim=-1) == 2).all()
    res_pred, atom_pred = split_feaures(pred_mask)
    res_target, atom_target = split_feaures(target)
    assert (res_target.sum(dim=-1) == 1).all()
    assert (atom_target.sum(dim=-1) == 1).all()
    loss_res = torch.nn.functional.cross_entropy(res_pred, res_target)
    loss_atom = torch.nn.functional.cross_entropy(atom_pred, atom_target)
    sr_res = success_rate(res_pred, res_target)
    sr_atom = success_rate(atom_pred, atom_target)
    return loss_res, loss_atom, sr_res, sr_atom


def learn(pocketfile, radius=6.0, batch_size=16, n_epochs=1000, device=None):
    if device is None:
        device = DEVICE
    print(f"Training on {device}")
    dataset = proteingraph.Dataset(
        txtfile=pocketfile, radius=radius, return_pyg_graph=True, masked_atom=True
    )
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gcn = messagepassing.GCN(n_n=58, n_e=1, n_o=256, embedding_dim=512, normalize=False)
    gcn = gcn.to(device)
    # optimizer = torch.optim.SGD(
    #     gcn.parameters(), lr=0.0001, momentum=0.9, nesterov=False, dampening=0.9
    # )
    optimizer = torch.optim.Adam(gcn.parameters())
    for epoch in range(n_epochs):
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            batch = batch.to(device)
            # DataBatch(x=[5188, 58], edge_index=[2, 65462], edge_attr=[65462, 1], y=[16], masked_features=[928], masked_atom_id=[16], batch=[5188], ptr=[17])
            out = gcn.forward(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch_index=batch.batch,
                return_node_features=True,
            )
            loss_res, loss_atom, sr_res, sr_atom = maskingLoss(out, batch)
            lossval = loss_res + loss_atom
            lossval.backward()
            optimizer.step()
            print(
                f"epoch: {epoch}|step: {i}|loss: {lossval:.4g}|loss_res: {loss_res:.4g}|loss_atom: {loss_atom:.4g}|sr_res: {sr_res:.4g}|sr_atom: {sr_atom:.4g}"
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
