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
import os
import wget
import datetime
import traceback
# ### UNCOMMENT FOR LOGGING ####
import logging

logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
logging.info(f"################ Starting {__file__} ################")
# ### ##################### ####


def collate_fn(batch):
    return batch


def download_uniprot_pdb():
    if not os.path.exists('data/uniprot_pdb.csv'):
        if not os.path.isdir('data'):
            os.mkdir('data')
        print('Downloading uniprot_pdb.csv')
        wget.download('http://ftp.ebi.ac.uk/pub/databases/msd/sifts/csv/uniprot_pdb.csv', 'data/uniprot_pdb.csv')


def read_uniprot_pdb(pdbpath, ext, exclude_list=None):
    """
    >>> np.random.seed(0)
    >>> list_IDs = read_uniprot_pdb(pdbpath='/media/bougui/scratch/pdb', ext='cif.gz')
    >>> len(list_IDs)
    49926
    >>> list_IDs
    array(['/media/bougui/scratch/pdb/1a/11as.cif.gz',
           '/media/bougui/scratch/pdb/54/154l.cif.gz',
           '/media/bougui/scratch/pdb/55/155c.cif.gz', ...,
           '/media/bougui/scratch/pdb/zz/6zzy.cif.gz',
           '/media/bougui/scratch/pdb/zz/6zzz.cif.gz',
           '/media/bougui/scratch/pdb/zz/7zzk.cif.gz'], dtype='<U40')
    """
    list_IDs = []
    n_excluded = 0
    exclude_list = set(exclude_list)
    with open('data/uniprot_pdb.csv', 'r') as file:
        for line in file:
            if line.startswith('#') or line.startswith('SP_PRIMARY'):
                continue
            # uniprot = line.split(',')[0]
            pdblist = line.strip().split(',')[1].split(';')
            if exclude_list is not None:
                n_ori = len(pdblist)
                pdblist = list(set(pdblist) - exclude_list)
                n_after = len(pdblist)
                n_excluded += n_ori - n_after
            if len(pdblist) > 0:
                pdb = np.random.choice(pdblist)
                # f'{pdbpath}/**/*.{ext}'
                list_IDs.append(f'{pdbpath}/{pdb[1:3]}/{pdb}.{ext}')
    list_IDs = np.unique(list_IDs)
    if exclude_list is not None:
        print(f"{n_excluded} data removed by exclude_list")
    return list_IDs


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

    Test with uniprot_pdb:
    >>> dataset = DensityDataset('/media/bougui/scratch/pdb', nsample=3, uniprot_pdb=True, list_ids_file='test.txt.gz')
    >>> d0 = dataset[0]
    >>> [e.shape for e in d0]
    [(71, 96, 69), (79, 93, 66), (68, 60, 59)]
    """
    def __init__(self,
                 pdbpath,
                 return_name=False,
                 nsample=1,
                 ext='cif.gz',
                 uniprot_pdb=False,
                 list_ids_file=None,
                 exclude_list=None,
                 verbose=False,
                 skip_error=False):
        """
        nsample: number of random sample (for rotations and chains) to get by system
        uniprot_pdb: download the list of uniprot for the pdb (if not present) and use it for loading
        exclude_list: list of pdbs to remove from the list
        """
        self.verbose = verbose
        self.skip_error = skip_error
        self.uniprot_pdb = uniprot_pdb
        if not self.uniprot_pdb:
            self.list_IDs = glob.glob(f'{pdbpath}/**/*.{ext}')
            if exclude_list is not None:
                exclude_list = [f'{pdbpath}/{e[1:3]}/{e}.{ext}' for e in exclude_list]
                print(exclude_list)
                n_ori = len(self.list_IDs)
                self.list_IDs = list(set(self.list_IDs) - set(exclude_list))
                n_after = len(self.list_IDs)
                n_excluded = n_ori - n_after
                print(f"Number of excluded pdb entries: {n_excluded}")
        else:
            download_uniprot_pdb()
            self.list_IDs = read_uniprot_pdb(pdbpath, ext, exclude_list=exclude_list)
        if list_ids_file is not None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bn, en = list_ids_file.split('.', 1)
            np.savetxt(f'{bn}_{ts}.{en}', self.list_IDs, fmt='%s')
        self.return_name = return_name
        self.nsample = nsample
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        if self.verbose:
            log(f'Density for {pdbfile}')
        sigma = np.random.uniform(1., 2.5)
        if self.nsample == 1:
            try:
                density, origin = Density(pdb=pdbfile,
                                          sigma=sigma,
                                          spacing=1,
                                          padding=(3, 3, 3),
                                          random_rotation=True,
                                          random_chains=True,
                                          obj=index)
            except Exception:
                if not self.skip_error:
                    raise
                else:
                    log(f'Error for {pdbfile}')
                    log(traceback.format_exc())
                    return None
            if self.verbose:
                log(f'Density for {pdbfile} done')
            return density
        else:
            densities = []
            for i in range(self.nsample):
                try:
                    density, origin = Density(pdb=pdbfile,
                                              sigma=sigma,
                                              spacing=1,
                                              padding=(3, 3, 3),
                                              random_rotation=True,
                                              random_chains=True)
                    densities.append(density)
                except Exception:
                    if not self.skip_error:
                        raise
                    else:
                        log(f'Error for {pdbfile} and view {i}')
                        log(traceback.format_exc())
                        return None
            if self.verbose:
                log(f'Densities for {pdbfile} done')
            return densities


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod()
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()
