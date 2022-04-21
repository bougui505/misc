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
from misc.protein.interpred import maps


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> # randomize must be set to True for real application. Just set to False for testing
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     if i == 12:
    ...         break
    >>> print(len(batch))
    4
    >>> [(A.shape, B.shape, cmap.shape) if A is not None else (A, B, cmap) for A, B, cmap in batch]
    [(None, None, None), (torch.Size([1, 1, 48, 3]), torch.Size([1, 1, 46, 3]), torch.Size([1, 1, 1, 48, 46])), (torch.Size([1, 1, 112, 3]), torch.Size([1, 1, 118, 3]), torch.Size([1, 1, 1, 112, 118])), (None, None, None)]

    In the example above, None is returned for protein with 1 chain only
    """
    def __init__(self, pdbpath, selection='polymer.protein and name CA', randomize=True):
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        self.randomize = randomize
        self.selection = selection
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        coords_a, coords_b, cmap = get_dimer(pymolname, self.selection, randomize=self.randomize)
        cmd.delete(pymolname)
        return coords_a, coords_b, cmap


def get_dimer(pymolname, selection, randomize=True):
    chains = cmd.get_chains(pymolname)
    if len(chains) <= 1:
        return None, None, None
    chain_coords = []
    for chain in chains:
        chain_coords.append(cmd.get_coords(selection=f'{pymolname} and {selection} and chain {chain}'))
    chain_coords = [e for e in chain_coords if e is not None and len(e) > 8]
    if len(chain_coords) <= 1:
        return None, None, None
    if randomize:
        random.shuffle(chain_coords)
    dobreak = False
    for i in range(len(chain_coords) - 1):
        A = torch.tensor(chain_coords[i][None, None, ...])
        for j in range(i + 1, len(chain_coords)):
            B = torch.tensor(chain_coords[j][None, None, ...])
            cmap = maps.get_inter_cmap(A, B)
            if cmap.sum() > 0.:
                dobreak = True
                break
        if dobreak:
            break
    if cmap.sum() > 0.:
        return A, B, cmap
    else:
        return None, None, None


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
