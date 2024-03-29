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
from misc.pytorch import torchify
import misc.protein.VAE.utils as utils
import numpy as np


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> # Generate random datapoints for testing
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', interpolate=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> for i, batch in enumerate(dataloader):
    ...     print(len(batch))
    ...     print([dmat.shape for dmat in batch])
    ...     if i == 0:
    ...         break
    4
    [torch.Size([1, 1, 249, 249]), torch.Size([1, 1, 639, 639]), torch.Size([1, 1, 390, 390]), torch.Size([1, 1, 131, 131])]

    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', interpolate=True)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> batch.shape
    torch.Size([4, 1, 224, 224])

    >>> dataset = PDBdataset(pdblistfile='pdbfilelist.txt', return_name=True, interpolate=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> len(batch)
    4
    >>> [(dmat.shape, name) for dmat, name in batch]
    [(torch.Size([1, 1, 162, 162]), 'pdb/00/pdb200l.ent.gz_A'), (torch.Size([1, 1, 154, 154]), 'pdb/01/pdb101m.ent.gz_A'), (torch.Size([1, 1, 154, 154]), 'pdb/02/pdb102m.ent.gz_A'), (torch.Size([1, 1, 163, 163]), 'pdb/02/pdb102l.ent.gz_A')]
    """
    def __init__(
            self,
            pdbpath=None,
            pdblist=None,
            pdblistfile=None,  # format: pdbfilename chain
            selection='polymer.protein and name CA',
            interpolate=True,
            interpolate_size=(224, 224),
            return_name=False):
        self.chain_from_list = False
        self.return_name = return_name
        if pdbpath is not None:
            self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        if pdblist is not None:
            self.list_IDs = pdblist
        if pdblistfile is not None:
            with open(pdblistfile, 'r') as filelist:
                self.list_IDs = filelist.read().splitlines()
            self.chain_from_list = True
        self.selection = selection
        self.interpolate = interpolate
        self.interpolate_size = interpolate_size
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        if not self.chain_from_list:
            pdbfile = self.list_IDs[index]
        else:
            pdbfile, chain = self.list_IDs[index].split()
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        chains = cmd.get_chains(f'{pymolname} and {self.selection}')
        nchains = len(chains)
        if nchains == 0:
            if self.return_name:
                return None, pdbfile
            else:
                return None
        else:
            if not self.chain_from_list:
                chain = np.random.choice(chains)
            coords = cmd.get_coords(selection=f'{pymolname} and {self.selection} and chain {chain}')
            if coords is None:
                if self.return_name:
                    return None, pdbfile + "_" + chain
                else:
                    return None
            coords = torch.tensor(coords[None, ...])
            dmat = utils.get_dmat(coords)
            if self.interpolate:
                dmat = torch.nn.functional.interpolate(dmat, size=self.interpolate_size)
                dmat = dmat[0]
        cmd.delete(pymolname)
        if self.return_name:
            return dmat, pdbfile + "_" + chain
        else:
            return dmat


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
