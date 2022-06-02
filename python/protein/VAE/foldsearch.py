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

import faiss
import PDBloader
import torch
import vae
import cmapvae
from pymol import cmd
import os
from misc import randomgen
import numpy as np


def foldsearch(pdbcode=None,
               pdblist=None,
               index=faiss.read_index('index.faiss/index.faiss'),
               ids=np.load('index.faiss/ids.npy'),
               model=cmapvae.load_model('models/cmapvae_20220525_0843.pt'),
               selection='all',
               batch_size=4,
               n_neighbors=5):
    """
    >>> index = faiss.read_index('index.faiss/index.faiss')
    >>> ids = np.load('index.faiss/ids.npy')
    >>> model = cmapvae.load_model('models/cmapvae_20220525_0843.pt')
    >>> foldsearch(pdbcode='1ycr', index=index, ids=ids, model=model, selection='chain A')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    if pdbcode is not None:
        try:
            os.mkdir('pdbfetch')
        except FileExistsError:
            pass
        pymolname = randomgen.randomstring()
        cmd.fetch(pdbcode, name=pymolname, file=f'pdbfetch/{pdbcode}.cif', type='cif')
        cmd.delete(pymolname)
        pdblist = [f'pdbfetch/{pdbcode}.cif']
    dataset = PDBloader.PDBdataset(pdblist=pdblist,
                                   interpolate=False,
                                   selection=f'polymer.protein and name CA and {selection}',
                                   return_name=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=PDBloader.collate_fn)
    for i, data in enumerate(dataloader):
        batch = [dmat.to(device) for dmat, name in data if dmat is not None]
        names = [name for dmat, name in data if dmat is not None]
        with torch.no_grad():
            _, latent_vectors = vae.forward_batch(batch, model, encode_only=True, sample=False)
        latent_vectors = latent_vectors.detach().cpu().numpy()
        Dmat, Imat = index.search(latent_vectors, n_neighbors)
        print_foldsearch_results(Imat, Dmat, names, ids)


def print_foldsearch_results(Imat, Dmat, query_names, ids):
    for ind, dist, query in zip(Imat, Dmat, query_names):
        result_pdb_list = ids[ind]
        print(f'query: {query}')
        for pdb, d in zip(result_pdb_list, dist):
            # print(pdb)  # pdb/hf/pdb4hfz.ent.gz_A
            pdbcode = os.path.basename(pdb)
            pdbcode = os.path.splitext(os.path.splitext(pdbcode)[0])[0][-4:]
            chain = pdb.split('_')[1]
            print(f'\t{pdbcode} {chain} {d:.4f}')


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='PDB code of the query')
    parser.add_argument('--pdbfile', help='pdb filename of the query')
    parser.add_argument('--index', help='path to the directory containing the index and ids', default='index.faiss')
    parser.add_argument('--model', help='VAE model to use', default='models/cmapvae_20220525_0843.pt')
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--sel', help='Selection for the query in pymol selection language', default='all')
    parser.add_argument('--n_neighbors', help='Number of neighbors to return (default:5)', default=5, type=int)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.pdbfile is None:
        pdblist = None
    else:
        pdblist = [args.pdbfile]
    index = faiss.read_index(f'{args.index}/index.faiss')
    ids = np.load(f'{args.index}/ids.npy')
    model = cmapvae.load_model(filename=args.model, latent_dims=args.latent_dims)
    foldsearch(pdbcode=args.pdb,
               pdblist=pdblist,
               index=index,
               ids=ids,
               selection=args.sel,
               n_neighbors=args.n_neighbors)
