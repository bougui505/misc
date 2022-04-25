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

    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     if i == 12:
    ...         break
    >>> print(len(batch))
    4
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(None, None, None, None), (torch.Size([1, 1, 58, 3]), torch.Size([1, 1, 44, 3]), torch.Size([1, 42, 58, 44]), torch.Size([1, 1, 1, 58, 44])), (torch.Size([1, 1, 116, 3]), torch.Size([1, 1, 107, 3]), torch.Size([1, 42, 116, 107]), torch.Size([1, 1, 1, 116, 107])), (None, None, None, None)]

    In the example above, None is returned for protein with 1 chain only

    Try with randomize set to True
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', randomize=True)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     if i == 12:
    ...         break
    >>> print(len(batch))
    4

    Try with a list of PDBs:
    >>> dataset = PDBdataset(pdblist=['data/1ycr.pdb'], randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> for batch in dataloader:
    ...     pass
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(torch.Size([1, 1, 85, 3]), torch.Size([1, 1, 13, 3]), torch.Size([1, 42, 85, 13]), torch.Size([1, 1, 1, 85, 13]))]
    """
    def __init__(self, pdbpath=None, pdblist=None, selection='polymer.protein and name CA', randomize=True):
        if pdbpath is not None:
            self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        if pdblist is not None:
            self.list_IDs = pdblist
        self.randomize = randomize
        self.selection = selection

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
        coords_a, coords_b, dmat, seq_a, seq_b = get_dimer(pymolname, self.selection, randomize=self.randomize)
        if coords_a is not None:
            assert len(
                seq_a) == coords_a.shape[2], f'seq_a is not of same length as coords_a ({len(seq_a)}, {coords_a.shape})'
            assert len(
                seq_b) == coords_b.shape[2], f'seq_b is not of same length as coords_b ({len(seq_b)}, {coords_b.shape})'
        cmd.delete(pymolname)
        if seq_a is not None:
            interseq = utils.get_inter_seq(seq_a, seq_b)
        else:
            interseq = None
        return coords_a, coords_b, interseq, dmat


def get_dimer(pymolname, selection, randomize=True):
    chains = cmd.get_chains(pymolname)
    if len(chains) <= 1:
        return None, None, None, None, None
    chain_coords = []
    chain_seqs = []
    for chain in chains:
        chain_coords.append(cmd.get_coords(selection=f'{pymolname} and {selection} and chain {chain}'))
        chain_seqs.append(utils.get_seq(pymolname, selection=f'{selection} and chain {chain}'))
    sel = [i for i, e in enumerate(chain_coords) if e is not None and len(e) > 8]
    chain_coords = [chain_coords[i] for i in sel]
    chain_seqs = [chain_seqs[i] for i in sel]
    nchains = len(chain_coords)
    log(f'nchains: {nchains}')
    log(f'chain_coords: {[len(e) for e in chain_coords]}')
    log(f'chain_seqs : {[len(e) for e in chain_seqs]}')
    if nchains <= 1:
        return None, None, None, None, None
    if randomize:
        order = np.random.choice(nchains, size=nchains, replace=False)
        chain_coords = [chain_coords[i] for i in order]
        chain_seqs = [chain_seqs[i] for i in order]
    dobreak = False
    for i in range(nchains - 1):
        A = torch.tensor(chain_coords[i][None, None, ...])
        seq_A = chain_seqs[i]
        for j in range(i + 1, nchains):
            B = torch.tensor(chain_coords[j][None, None, ...])
            seq_B = chain_seqs[j]
            dmat = utils.get_inter_dmat(A, B)
            if dmat.min() < 8.:
                dobreak = True
                break
        if dobreak:
            break
    if dmat.min() < 8.:
        return A, B, dmat, seq_A, seq_B
    else:
        return None, None, None, None, None


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
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
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
