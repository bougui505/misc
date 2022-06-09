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

import misc.protein.sscl.encoder as encoder
import misc.protein.sscl.utils as utils
import torch
import PDBloader
import faiss
import os
from misc.eta import ETA
import time
import numpy as np
import datetime


def encode_pdb(pdb, sel='all', model='models/sscl_20220609_1344.pt', latent_dims=512):
    """
    >>> z = encode_pdb('pdb/yc/pdb1ycr.ent.gz')
    >>> z.shape
    torch.Size([1, 512])
    """
    coords = utils.get_coords(pdb, sel=sel)
    dmat = utils.get_dmat(coords[None, ...])
    model = encoder.load_model(model, latent_dims=latent_dims)
    with torch.no_grad():
        z = model(dmat)
    return z


def get_latent_similarity(pdb1, pdb2, sel1='all', sel2='all', model='models/sscl_20220609_1344.pt', latent_dims=512):
    """
    >>> pdb1 = 'pdb/yc/pdb1ycr.ent.gz'
    >>> pdb2 = 'pdb/yc/pdb1ycr.ent.gz'
    >>> get_latent_similarity(pdb1, pdb2, sel1='chain A', sel2='chain A and resi 25-109')
    1.0000...
    >>> get_latent_similarity(pdb1, pdb2, sel1='chain A', sel2='chain A and resi 25-64')
    0.9419...
    """
    z1 = encode_pdb(pdb1, model=model, latent_dims=latent_dims, sel=sel1)
    z2 = encode_pdb(pdb2, model=model, latent_dims=latent_dims, sel=sel2)
    sim = float(torch.matmul(z1, z2.T).squeeze().numpy())
    return sim


def build_index(pdblistfile,
                model,
                latent_dims=512,
                batch_size=4,
                do_break=np.inf,
                save_each=10,
                indexfilename='index.faiss'):
    """
    >>> model = encoder.load_model('models/sscl_20220609_1344.pt')
    >>> build_index('pdbfilelist.txt', model, do_break=3)
    Total number of pdb in the FAISS index: 12
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        os.mkdir(indexfilename)
    except FileExistsError:
        pass
    dataset = PDBloader.PDBdataset(pdblistfile=pdblistfile, return_name=True, do_fragment=False)
    model = model.to(device)
    model.eval()
    index = faiss.IndexFlatIP(latent_dims)  # build the index in cosine similarity (IP: Inner Product)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    names = []
    eta = ETA(total_steps=len(dataloader))
    t_0 = time.time()
    for i, data in enumerate(dataloader):
        if i >= do_break:
            break
        batch = [dmat.to(device) for dmat, name in data if dmat is not None]
        names.extend([name for dmat, name in data if dmat is not None])
        with torch.no_grad():
            latent_vectors = torch.cat([model(e) for e in batch])
        # print(latent_vectors.shape)  # torch.Size([4, 512])
        latent_vectors = latent_vectors.detach().cpu().numpy()
        index.add(latent_vectors)
        eta_val = eta(i + 1)
        if (time.time() - t_0) / 60 >= save_each:
            t_0 = time.time()
            faiss.write_index(index, f'{indexfilename}/index.faiss')
            np.save(f'{indexfilename}/ids.npy', np.asarray(names))
        last_saved = (time.time() - t_0)
        last_saved = str(datetime.timedelta(seconds=last_saved))
        log(f"step: {i+1}|last_saved: {last_saved}|eta: {eta_val}")
    npdb = index.ntotal
    print(f'Total number of pdb in the FAISS index: {npdb}')
    faiss.write_index(index, f'{indexfilename}/index.faiss')
    names = np.asarray(names)
    np.save(f'{indexfilename}/ids.npy', names)


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
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--similarity',
                        help='Compute the latent similarity between the 2 given pdb',
                        nargs=2,
                        metavar='file.pdb')
    parser.add_argument('-q', '--query', help='Search for nearest neighbors for the given query pdb')
    parser.add_argument('-n', help='Number of neighbors to return', type=int, default=5)
    parser.add_argument('--sel',
                        help='Selection for pdb file. Give two selections for the similarity computation',
                        default=None,
                        nargs='+')
    parser.add_argument('--model', help='SSCL model to use', metavar='model.pt', default='models/sscl_20220609_1344.pt')
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--build_index', help='Build the FAISS index', action='store_true')
    parser.add_argument('--save_every',
                        help='Save the FAISS index every given number of minutes when building it',
                        type=int,
                        default=10)
    parser.add_argument(
        '--pdblist',
        help='File containing the list of pdb file to put in the index. Line format of the file: pdbfile chain')
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--index', help='FAISS index directory. Default: index.faiss', default='index.faiss')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    if args.similarity is not None:
        pdb1 = args.similarity[0]
        pdb2 = args.similarity[1]
        if args.sel is None:
            sel1 = 'all'
            sel2 = 'all'
        else:
            sel1 = args.sel[0]
            sel2 = args.sel[1]
        sim = get_latent_similarity(pdb1, pdb2, sel1=sel1, sel2=sel2, model=args.model, latent_dims=args.latent_dims)
        print(f'similarity: {sim:.4f}')
    if args.query is not None:
        if args.sel is None:
            sel = 'all'
        else:
            sel = args.sel[0]
        z = encode_pdb(pdb=args.query, sel=sel, model=args.model, latent_dims=args.latent_dims)
        z = z.detach().cpu().numpy()
        index = faiss.read_index(f'{args.index}/index.faiss')
        ids = np.load(f'{args.index}/ids.npy')
        Dmat, Imat = index.search(z, args.n)
        print_foldsearch_results(Imat, Dmat, [args.query], ids)
    if args.build_index:
        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        build_index(args.pdblist,
                    model,
                    latent_dims=args.latent_dims,
                    batch_size=args.bs,
                    save_each=args.save_every,
                    indexfilename=args.index)
