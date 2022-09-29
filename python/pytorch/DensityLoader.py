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
from misc.protein.density import Density
import numpy as np


def collate_fn(batch):
    return batch


class DensityDataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> np.random.seed(seed=0)
    >>> seed = torch.manual_seed(0)
    >>> # Generate random datapoints for testing
    >>> dataset = DensityDataset('/media/bougui/scratch/pdb')
    >>> len(dataset)
    195858
    >>> dataset[0].shape
    (47, 44, 63)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> for i, batch in enumerate(dataloader):
    ...     print(len(batch))
    ...     if i == 0:
    ...         break
    4
    >>> [d.shape for d in batch]
    [(52, 32, 60), (69, 70, 84), (41, 42, 53), (66, 121, 58)]

    # For nsample > 1:
    >>> dataset = DensityDataset('/media/bougui/scratch/pdb', nsample=3)
    >>> d0 = dataset[0]
    >>> [e.shape for e in d0]
    [(46, 53, 51), (56, 43, 54), (35, 66, 41)]
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
    >>> for i, batch in enumerate(dataloader):
    ...     print(len(batch))
    ...     if i == 0:
    ...         break
    4
    >>> [len(l) for l in batch]
    [3, 3, 3, 3]
    >>> [[e.shape for e in l] for l in batch]
    [[(60, 50, 40), (52, 51, 50), (41, 58, 55)], [(97, 71, 65), (137, 91, 94), (86, 58, 77)], [(46, 53, 38), (74, 60, 71), (89, 52, 66)], [(65, 144, 66), (45, 49, 59), (71, 137, 63)]]
    """
    def __init__(self, pdbpath, return_name=False, nsample=1, ext='cif.gz'):
        """
        nsample: number of random sample (for rotations and chains) to get by system
        """
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.{ext}')
        self.return_name = return_name
        self.nsample = nsample
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        sigma = np.random.uniform(1., 2.5)
        if self.nsample == 1:
            density, origin = Density(pdb=pdbfile,
                                      sigma=sigma,
                                      spacing=1,
                                      padding=(3, 3, 3),
                                      random_rotation=True,
                                      random_chains=True)
            return density
        else:
            densities = []
            for i in range(self.nsample):
                density, origin = Density(pdb=pdbfile,
                                          sigma=sigma,
                                          spacing=1,
                                          padding=(3, 3, 3),
                                          random_rotation=True,
                                          random_chains=True)
                densities.append(density)
            return densities


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
