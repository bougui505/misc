#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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
import glob
from pymol import cmd
from misc import randomgen
import random
from misc.protein.interpred import utils
import numpy as np


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> # randomize must be set to True for real application. Just set to False for testing

    >>> dataset = PDBdataset('/media/bougui/scratch/dimerdb')
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     if i == 12:
    ...         break
    >>> print(len(batch))
    4
    >>> [(inp.shape, intercmap.shape) for (inp, intercmap) in batch]
    [(torch.Size([1, 1, 21, 28]), torch.Size([1, 1, 21, 28])), (torch.Size([1, 1, 245, 242]), torch.Size([1, 1, 245, 242])), (torch.Size([1, 1, 28, 28]), torch.Size([1, 1, 28, 28])), (torch.Size([1, 1, 188, 188]), torch.Size([1, 1, 188, 188]))]

    Try with a list of PDBs:
    >>> dataset = PDBdataset(pdblist=['data/1ycr.pdb'], randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> for batch in dataloader:
    ...     pass
    >>> print(len(batch))
    1
    >>> [(inp.shape, intercmap.shape) for (inp, intercmap) in batch]
    [(torch.Size([1, 1, 85, 13]), torch.Size([1, 1, 85, 13]))]
    """
    def __init__(self,
                 pdbpath=None,
                 pdblist=None,
                 selection='polymer.protein and name CA',
                 randomize=True,
                 return_name=False):
        if pdbpath is not None:
            self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        if pdblist is not None:
            self.list_IDs = pdblist
        self.randomize = randomize
        self.selection = selection
        self.return_name = return_name

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        log(f'pdbfile: {pdbfile}')
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        # Remove alternate locations (see: https://pymol.org/dokuwiki/doku.php?id=concept:alt)
        cmd.remove("not alt ''+A")
        cmd.alter(pymolname, "alt=''")
        ######################################################################################
        coords_a, coords_b, cmap, seq_a, seq_b = get_dimer(pymolname, self.selection, randomize=self.randomize)
        cmd.delete(pymolname)
        log(f'coords_a.shape: {coords_a.shape}')
        inp = utils.get_input(coords_a, coords_b)
        if self.return_name:
            return inp, cmap, pdbfile
        else:
            return inp, cmap


def get_dimer(pymolname, selection, randomize=True):
    chains = cmd.get_chains(f'{pymolname} and {selection}')
    nchains = len(chains)
    log(f'nchains: {nchains}')
    assert nchains == 2, f'The number of chains ({nchains}) is not 2'
    chain_coords = []
    chain_seqs = []
    for chain in chains:
        chain_coords.append(cmd.get_coords(selection=f'{pymolname} and {selection} and chain {chain}'))
        chain_seqs.append(utils.get_seq(pymolname, selection=f'{selection} and chain {chain}'))
    log(f'nchains: {nchains}')
    log(f'chain_coords: {[len(e) for e in chain_coords]}')
    log(f'chain_seqs : {[len(e) for e in chain_seqs]}')
    A, B = [torch.Tensor(c) for c in chain_coords]
    seq_A, seq_B = chain_seqs
    cmap = utils.get_inter_cmap(A[None, ...], B[None, ...])
    return A, B, cmap, seq_A, seq_B


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    from misc.eta import ETA
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
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--load', help='Load a db for testing')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
    if args.load is not None:
        num_workers = os.cpu_count()
        dataset = PDBdataset(pdbpath=args.load)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 collate_fn=collate_fn)
        eta = ETA(len(dataloader))
        for i, _ in enumerate(dataloader):
            eta_val = eta(i + 1)
            log(f"step: {i}|eta: {eta_val}")
