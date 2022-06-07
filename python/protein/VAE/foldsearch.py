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
import scipy.spatial.distance as scidist
from utils import Normalizer


def foldsearch(pdbcode=None,
               pdblist=None,
               index=faiss.read_index('index_20220603_1045.faiss/index.faiss'),
               ids=np.load('index_20220603_1045.faiss/ids.npy'),
               model=cmapvae.load_model('models/cmapvae_20220525_0843.pt'),
               selection='all',
               batch_size=4,
               n_neighbors=5,
               return_latent=False,
               print_latent=False,
               normalize=True):
    """
    >>> index = faiss.read_index('index_20220603_1045.faiss/index.faiss')
    >>> ids = np.load('index_20220603_1045.faiss/ids.npy')
    >>> model = cmapvae.load_model('models/cmapvae_20220525_0843.pt')
    >>> foldsearch(pdbcode=['1ycr'], index=index, ids=ids, model=model, selection='chain A')
    query: pdbfetch/1ycr.cif_A
        1ycr A 1.0000
        5vk0 G 0.9993
        3lnz A 0.9993
        6t2f A 0.9992
        6t2e A 0.9989
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    if pdbcode is not None:
        try:
            os.mkdir('pdbfetch')
        except FileExistsError:
            pass
        pdblist = []
        for pdb in pdbcode:
            pymolname = randomgen.randomstring()
            cmd.fetch(pdb, name=pymolname, file=f'pdbfetch/{pdb}.cif', type='cif')
            cmd.delete(pymolname)
            pdblist.append(f'pdbfetch/{pdb}.cif')
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
        if normalize:
            normalizer = Normalizer(batch)
            batch = normalizer.transform(normalizer.batch)
        names = [name for dmat, name in data if dmat is not None]
        with torch.no_grad():
            _, latent_vectors = vae.forward_batch(batch, model, encode_only=True, sample=False)
        latent_vectors = latent_vectors.detach().cpu().numpy()
        latent_vectors = latent_vectors / np.linalg.norm(latent_vectors, axis=1)[:, None]
        if return_latent:
            if print_latent:
                print_vector(latent_vectors[0])
            return latent_vectors
        Dmat, Imat = index.search(latent_vectors, n_neighbors)
        print_foldsearch_results(Imat, Dmat, names, ids)


def get_distance(pdbcode=None,
                 pdblist=None,
                 index=faiss.read_index('index_20220603_1045.faiss/index.faiss'),
                 ids=np.load('index_20220603_1045.faiss/ids.npy'),
                 model=cmapvae.load_model('models/cmapvae_20220525_0843.pt'),
                 selection=None):
    """
    >>> index = faiss.read_index('index_20220603_1045.faiss/index.faiss')
    >>> ids = np.load('index_20220603_1045.faiss/ids.npy')
    >>> model = cmapvae.load_model('models/cmapvae_20220525_0843.pt')
    >>> get_distance(pdbcode=['1ycr', '5vk0'], index=index, ids=ids, model=model, selection=['chain A', 'chain G'])
    array([0.99931141])
    """
    latent_vectors = []
    for sel, pdb in zip(selection, pdbcode):
        v = foldsearch(pdbcode=[pdb],
                       pdblist=pdblist,
                       index=index,
                       ids=ids,
                       model=model,
                       selection=sel,
                       return_latent=True)
        latent_vectors.append(np.squeeze(v))
    latent_vectors = np.asarray(latent_vectors)
    # print(latent_vectors.shape)
    # (2, 512)
    pdist = 1. - scidist.pdist(latent_vectors, metric='cosine')
    return pdist


def print_foldsearch_results(Imat, Dmat, query_names, ids):
    for ind, dist, query in zip(Imat, Dmat, query_names):
        result_pdb_list = ids[ind]
        print(f'query: {query}')
        for pdb, d in zip(result_pdb_list, dist):
            # print(pdb)  # pdb/hf/pdb4hfz.ent.gz_A
            pdbcode = os.path.basename(pdb)
            pdbcode = os.path.splitext(os.path.splitext(pdbcode)[0])[0][-4:]
            chain = pdb.split('_')[1]
            print(f'    {pdbcode} {chain} {d:.4f}')


def print_vector(v):
    print(' '.join([f'{e:.4f}' for e in v]))


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
    parser.add_argument(
        '--pdb',
        help='PDB code of the query. If 2 or more PDB codes are given compute the distance between those PDBs',
        nargs='+')
    parser.add_argument('--pdbfile', help='pdb filename of the query')
    parser.add_argument('--index',
                        help='path to the directory containing the index and ids',
                        default='index_20220603_1045.faiss')
    parser.add_argument('--model', help='VAE model to use', default='models/cmapvae_20220525_0843.pt')
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--sel', help='Selection for the query in pymol selection language', default='all', nargs='+')
    parser.add_argument('--print_latent', help='Print on stdout the latent vector of the query', action='store_true')
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
    if len(args.pdb) == 1:
        foldsearch(pdbcode=args.pdb,
                   pdblist=pdblist,
                   index=index,
                   ids=ids,
                   selection=args.sel[0],
                   n_neighbors=args.n_neighbors,
                   return_latent=args.print_latent,
                   print_latent=args.print_latent)
    else:
        pdist = get_distance(pdbcode=args.pdb, pdblist=pdblist, index=index, ids=ids, model=model, selection=args.sel)
        print_vector(pdist)
