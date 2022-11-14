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
import misc.protein.sscl.utils as utils
import numpy as np

cmd.feedback(action='disable', module='all', mask='everything')


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> seed = torch.manual_seed(0)
    >>> dataset = PDBdataset(pdbpath='/media/bougui/scratch/pdb', return_name=True)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> len(batch)
    4
    >>> for e in batch:
    ...     if e[0] is not None:
    ...         dmat, dmat_fragment, name = e
    ...         print(dmat.shape, dmat_fragment.shape, name)
    torch.Size([1, 1, 589, 589]) torch.Size([1, 1, 875, 875]) /media/bougui/scratch/pdb/a9/3a9r.cif.gz_C
    torch.Size([1, 1, 116, 116]) torch.Size([1, 1, 199, 199]) /media/bougui/scratch/pdb/a9/4a9j.cif.gz_B
    torch.Size([1, 1, 198, 198]) torch.Size([1, 1, 260, 260]) /media/bougui/scratch/pdb/a9/6a92.cif.gz_C
    """
    def __init__(
        self,
        pdbpath=None,
        pdblist=None,
        pdblistfile=None,  # format: pdbfilename chain
        selection='polymer.protein and name CA',
        return_name=False,
        do_fragment=True,
        ext='cif.gz'):
        self.chain_from_list = False
        self.return_name = return_name
        self.do_fragment = do_fragment
        if pdbpath is not None:
            self.list_IDs = glob.glob(f'{pdbpath}/**/*.{ext}')
        if pdblist is not None:
            self.list_IDs = pdblist
        if pdblistfile is not None:
            with open(pdblistfile, 'r') as filelist:
                self.list_IDs = filelist.read().splitlines()
            self.chain_from_list = True
        self.selection = selection
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
            if self.do_fragment:
                fragment = utils.get_random_fragment(coords)
                dmat_fragment = utils.get_dmat(fragment)
            else:
                dmat_fragment = None
        cmd.delete(pymolname)
        if self.return_name:
            if dmat_fragment is not None:
                return dmat, dmat_fragment, pdbfile + "_" + chain
            else:
                return dmat, pdbfile + "_" + chain
        else:
            if dmat_fragment is not None:
                return dmat, dmat_fragment
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
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
