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
from misc.protein.interpred import utils
from misc.protein.interpred import PDBloader
import os
from misc.eta import ETA


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

    >>> interpred = InterPred()
    >>> interpred(coords_A, coords_B, interseq).shape
    torch.Size([1, 85, 13])
    >>> len(list(interpred.parameters()))
    18
    """
    def __init__(self):
        super(InterPred, self).__init__()
        self.fcn_a = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding='same'))
        self.fcn_b = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding='same'),
                                         torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding='same'))
        self.fcn_seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=42, out_channels=16, kernel_size=3, padding='same'),
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same'),
            torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding='same'))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, coords_a, coords_b, interseq):
        # batchsize, na, spacedim = coords_a.shape
        dmat_a = utils.get_dmat(coords_a)
        dmat_b = utils.get_dmat(coords_b)
        out_a = self.fcn_a(dmat_a)
        out_a = out_a.mean(axis=-1)
        out_b = self.fcn_a(dmat_b)
        out_b = out_b.mean(axis=-1)
        out = torch.einsum('ijk,lmn->ikn', out_a, out_b)
        out_seq = self.fcn_seq(interseq)[:, 0, :, :]
        out = out * out_seq
        out = self.sigmoid(out)
        return out


def learn(pdbpath=None, pdblist=None, nepoch=10, batch_size=4, num_workers=None, print_each=100):
    """
    >>> learn(pdblist=['data/1ycr.pdb'], print_each=1, nepoch=100)
    """
    interpred = InterPred()
    optimizer = torch.optim.Adam(interpred.parameters())
    if num_workers is None:
        num_workers = os.cpu_count()
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist, randomize=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    epoch = 0
    step = 0
    eta = ETA(total_steps=nepoch * len(dataiter))
    while epoch < nepoch:
        try:
            batch = next(dataiter)
            out, targets = forward_batch(batch, interpred)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss = get_loss(out, targets)
            loss.backward()
            optimizer.step()
            step += 1
            if not step % print_each:
                eta_val = eta(step)
                log(f"epoch: {epoch+1}|step: {step}|loss: {loss}|eta: {eta_val}")
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1


def forward_batch(batch, interpred):
    """
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(None, None, None, None), (torch.Size([1, 1, 639, 3]), torch.Size([1, 1, 639, 3]), torch.Size([1, 42, 639, 639]), torch.Size([1, 1, 1, 639, 639])), (torch.Size([1, 1, 390, 3]), torch.Size([1, 1, 390, 3]), torch.Size([1, 42, 390, 390]), torch.Size([1, 1, 1, 390, 390])), (None, None, None, None)]
    >>> interpred = InterPred()
    >>> out, targets = forward_batch(batch, interpred)
    >>> len(out)
    2
    >>> [e.shape for e in out]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> [e.shape for e in targets]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    """
    out = []
    targets = []
    for data in batch:
        coords_a, coords_b, interseq, cmap = data
        if coords_a is not None:
            coords_a = coords_a[0]  # Remove the extra dimension not required
            coords_b = coords_b[0]
            intercmap = interpred(coords_a, coords_b, interseq)
            out.append(intercmap)
            targets.append(cmap[0, 0, ...])
    return out, targets


def get_loss(out_batch, targets):
    """
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(None, None, None, None), (torch.Size([1, 1, 639, 3]), torch.Size([1, 1, 639, 3]), torch.Size([1, 42, 639, 639]), torch.Size([1, 1, 1, 639, 639])), (torch.Size([1, 1, 390, 3]), torch.Size([1, 1, 390, 3]), torch.Size([1, 42, 390, 390]), torch.Size([1, 1, 1, 390, 390])), (None, None, None, None)]
    >>> interpred = InterPred()

    >>> out_batch, targets = forward_batch(batch, interpred)
    >>> len(out_batch)
    2
    >>> [e.shape for e in out_batch]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> [e.shape for e in targets]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> loss = get_loss(out_batch, targets)
    >>> loss
    tensor(..., grad_fn=<DivBackward0>)
    """
    n = len(out_batch)
    loss = 0.
    for i in range(n):
        inp = out_batch[i].float()
        target = targets[i].float()
        loss += torch.nn.functional.binary_cross_entropy(inp.flatten()[None, ...], target.flatten()[None, ...])
    loss = loss / n
    return loss


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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
