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
import misc.protein.get_pdb_title as get_pdb_title
import torch
import PDBloader
import faiss
import os
from misc.eta import ETA
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt


def encode_pdb(pdb, model, sel='all', latent_dims=512, return_dmat=False):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> z, conv = encode_pdb('pdb/yc/pdb1ycr.ent.gz', model)
    >>> z.shape
    torch.Size([1, 512])
    >>> conv.shape
    torch.Size([1, 512, 98, 98])
    """
    coords = utils.get_coords(pdb, sel=sel)
    dmat = utils.get_dmat(coords[None, ...])
    with torch.no_grad():
        try:
            # FCN model
            z, conv = model(dmat, get_conv=True)
        except TypeError:
            # CNN model
            z = model(dmat)
            conv = None
    if return_dmat:
        return z, conv, dmat
    else:
        return z, conv


def get_latent_similarity(pdb1, pdb2, model, sel1='all', sel2='all', latent_dims=512, doplot=False):
    """
    >>> pdb1 = 'pdb/yc/pdb1ycr.ent.gz'
    >>> pdb2 = 'pdb/yc/pdb1ycr.ent.gz'
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> sim, feature_importance = get_latent_similarity(pdb1, pdb2, model, sel1='chain A', sel2='chain A and resi 25-109')
    >>> sim
    1.0000...
    >>> sim, feature_importance = get_latent_similarity(pdb1, pdb2, model, sel1='chain A', sel2='chain A and resi 25-64', doplot=True)
    >>> sim
    0.9474...
    """
    z1, conv1, dmat1 = encode_pdb(pdb1, model=model, latent_dims=latent_dims, sel=sel1, return_dmat=True)
    z2, conv2, dmat2 = encode_pdb(pdb2, model=model, latent_dims=latent_dims, sel=sel2, return_dmat=True)
    dmat1 = dmat1.detach().cpu().numpy().squeeze()
    dmat2 = dmat2.detach().cpu().numpy().squeeze()
    sim = float(torch.matmul(z1, z2.T).squeeze().numpy())
    feature_importance = (z1 * z2).squeeze().numpy() / sim
    # f1 = get_feature_map(conv1, feature_importance)
    # f2 = get_feature_map(conv2, feature_importance)
    if doplot:
        plot_sim(dmat1, dmat2, conv1, conv2, z1, z2, sim)
    return sim, feature_importance


def plot_sim(dmat1, dmat2, conv1, conv2, z1, z2, sim, threshold=8.):
    fig = plt.figure(constrained_layout=True)

    def onclick(event):
        ind = int(event.xdata)
        maxconv1 = conv1[ind, ...].max()
        maxconv2 = conv2[ind, ...].max()
        _, i1, j1 = np.unravel_index(conv1[ind, ...].argmax(), conv1.shape)
        _, i2, j2 = np.unravel_index(conv2[ind, ...].argmax(), conv2.shape)
        print(ind, z1[ind], z2[ind], maxconv1, maxconv2)
        # ax2.matshow(conv1[ind, ...] == maxconv1)
        # ax3.matshow(conv2[ind, ...] == maxconv2)
        ax0.scatter(j1, i1)
        ax1.scatter(j2, i2)
        ax4.scatter(ind, z1[ind], zorder=2)
        fig.canvas.draw_idle()

    z1 = z1.detach().cpu().numpy().squeeze()
    z2 = z2.detach().cpu().numpy().squeeze()
    conv1 = conv1.cpu().detach().numpy().squeeze()
    conv2 = conv2.cpu().detach().numpy().squeeze()
    cmap1 = utils.get_cmap(torch.tensor(dmat1))
    cmap2 = utils.get_cmap(torch.tensor(dmat2))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    spec = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(spec[0, 0])
    ax0.matshow(cmap1)
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.matshow(cmap2)
    # ax2 = fig.add_subplot(spec[1, 0])
    # ax2.matshow(conv1[0, ...])
    # ax3 = fig.add_subplot(spec[1, 1])
    # ax3.matshow(conv2[0, ...])
    ax4 = fig.add_subplot(spec[1, :])
    ax4.plot(z1)
    ax4.plot(z2)
    ax4.set_xlabel(f'latent_space: sim={sim:.4f}')
    plt.show()


def maxpool(conv):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> z, conv = encode_pdb('pdb/yc/pdb1ycr.ent.gz', model)
    >>> conv.shape
    torch.Size([1, 512, 98, 98])
    >>> out = maxpool(conv)
    >>> out.shape
    (1, 512, 98, 98)
    """
    conv = conv.detach().cpu().numpy()
    n = conv.shape[-1]
    latent_dim = conv.shape[1]
    bs = conv.shape[0]
    conv = conv.reshape((bs, latent_dim, -1))
    inds = conv.argmax(axis=-1)
    # print(inds.shape)  # (1, 512)
    out = np.zeros((bs, latent_dim, n * n))
    for b in range(bs):
        for la in range(latent_dim):
            i = inds[b, la]
            out[b, la, i] = conv[b, la, i]
    out = out
    out = out.reshape((bs, latent_dim, n, n))
    return out


def get_important_features(feature_importance, threshold=0.5):
    """
    >>> pdb1 = 'pdb/ay/pdb7aye.ent.gz'
    >>> pdb2 = 'pdb/di/pdb4dij.ent.gz'
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> sim, feature_importance = get_latent_similarity(pdb1, pdb2, model, sel1='chain A', sel2='chain A')
    >>> sim
    0.9953...
    >>> feature_importance.shape
    (512,)
    >>> feature_importance.sum()
    1.0
    >>> get_important_features(feature_importance)
    [(146, 0.16278093), (9, 0.13784872), (237, 0.09079589), (37, 0.03166283), (502, 0.027908802), (277, 0.024614722), (67, 0.019103186)]
    """
    important_features = feature_importance.argsort()[::-1]
    csum = np.cumsum(feature_importance[important_features])
    important_features = important_features[csum <= threshold]
    importances = feature_importance[important_features]
    return list(zip(important_features, importances))


def get_feature_map(conv, feature_importance):
    """
    >>> pdb1 = 'pdb/yc/pdb1ycr.ent.gz'
    >>> pdb2 = 'pdb/rv/pdb1rv1.ent.gz'
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> sim, feature_importance = get_latent_similarity(pdb1, pdb2, model, sel1='chain A', sel2='chain A')
    >>> sim
    0.9974226951599121
    >>> feature_importance.shape
    (512,)
    >>> feature_importance.sum()
    0.99999994
    >>> important_features = get_important_features(feature_importance, threshold=0.5)
    >>> z, conv = encode_pdb(pdb1, model)
    >>> conv.shape
    torch.Size([1, 512, 98, 98])
    >>> out = get_feature_map(conv, feature_importance)
    >>> out.shape
    (98, 98)

    # >>> _ = plt.matshow(out)
    # >>> _ = plt.colorbar()
    # >>> plt.show()
    """
    # conv = maxpool(conv).squeeze()
    conv = conv.detach().cpu().numpy().squeeze()
    conv -= conv.min(axis=0)
    conv /= conv.max(axis=0)
    conv = conv * feature_importance[:, None, None]
    out = conv.sum(axis=0)
    # out = (out + out.T) / 2.
    return out


def build_index(pdblistfile,
                model,
                latent_dims=512,
                batch_size=4,
                do_break=np.inf,
                save_each=10,
                indexfilename='index.faiss'):
    """
    >>> model = encoder.load_model('models/sscl_fcn_20220610_1353.pt')
    Loading FCN model
    >>> build_index('pdbfilelist.txt', model, do_break=3, indexfilename='index_test.faiss')
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
        try:
            with torch.no_grad():
                latent_vectors = torch.cat([model(e) for e in batch])
            # print(latent_vectors.shape)  # torch.Size([4, 512])
            latent_vectors = latent_vectors.detach().cpu().numpy()
            index.add(latent_vectors)
            names.extend([name for dmat, name in data if dmat is not None])
        except RuntimeError:
            print('System too large to fit in memory:')
            print([name for dmat, name in data if dmat is not None])
            print([dmat.shape for dmat, name in data if dmat is not None])
            pass
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


def print_foldsearch_results(Imat, Dmat, query_names, ids, return_name=False):
    for ind, dist, query in zip(Imat, Dmat, query_names):
        result_pdb_list = ids[ind]
        print(f'query: {query}')
        for pdb, d in zip(result_pdb_list, dist):
            # print(pdb)  # pdb/hf/pdb4hfz.ent.gz_A
            pdbcode = os.path.basename(pdb)
            pdbcode = os.path.splitext(os.path.splitext(pdbcode)[0])[0][-4:]
            chain = pdb.split('_')[1]
            outstr = f'>>> {pdbcode} {chain} {d:.4f}'
            if return_name:
                title = get_pdb_title.get_pdb_title(pdbcode, chain)
                outstr += f' {title}'
            print(outstr)


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
    parser.add_argument('--title', help='Retrieve PDB title information', action='store_true')
    parser.add_argument('-n', help='Number of neighbors to return', type=int, default=5)
    parser.add_argument('--sel',
                        help='Selection for pdb file. Give two selections for the similarity computation',
                        default=None,
                        nargs='+')
    parser.add_argument('--model',
                        help='SSCL model to use',
                        metavar='model.pt',
                        default='models/sscl_fcn_20220614_0927.pt')
    parser.add_argument('--latent_dims', default=128, type=int)
    parser.add_argument('--build_index', help='Build the FAISS index', action='store_true')
    parser.add_argument('--save_every',
                        help='Save the FAISS index every given number of minutes when building it',
                        type=int,
                        default=10)
    parser.add_argument(
        '--pdblist',
        help='File containing the list of pdb file to put in the index. Line format of the file: pdbfile chain')
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--index',
                        help='FAISS index directory. Default: index_fcn_20220610_1353.faiss',
                        default='index_fcn_20220614_0927.faiss')
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

        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        sim, feature_importance = get_latent_similarity(pdb1,
                                                        pdb2,
                                                        sel1=sel1,
                                                        sel2=sel2,
                                                        model=model,
                                                        latent_dims=args.latent_dims,
                                                        doplot=True)
        print(f'similarity: {sim:.4f}')
    if args.query is not None:
        if args.sel is None:
            sel = 'all'
        else:
            sel = args.sel[0]
        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        z, conv = encode_pdb(pdb=args.query, sel=sel, model=model, latent_dims=args.latent_dims)
        z = z.detach().cpu().numpy()
        index = faiss.read_index(f'{args.index}/index.faiss')
        ids = np.load(f'{args.index}/ids.npy')
        Dmat, Imat = index.search(z, args.n)
        print_foldsearch_results(Imat, Dmat, [args.query], ids, return_name=args.title)
    if args.build_index:
        model = encoder.load_model(args.model, latent_dims=args.latent_dims)
        build_index(args.pdblist,
                    model,
                    latent_dims=args.latent_dims,
                    batch_size=args.bs,
                    save_each=args.save_every,
                    indexfilename=args.index)
