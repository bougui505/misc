#!/usr/bin/env python
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
import os
import torch
from misc.pytorch import DensityLoader, contrastive_loss, resnet3d, trainer
import numpy as np
from tqdm import tqdm
from misc.protein.density import Density
import glob
from misc.annoy.NNindex import NNindex
from misc.Timer import Timer
from misc.Grid3 import mrc

TIMER = Timer(autoreset=True)

LOSS = contrastive_loss.SupConLoss(temperature=0.2, base_temperature=0.2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def loss_function(batch, out):
    """
    >>> seed = torch.manual_seed(0)
    >>> bsz = 10  # Batch size
    >>> n_views = 5  # the number of crops from each image (positive example)
    >>> latent_dim = 256
    >>> features = torch.randn(bsz, n_views, latent_dim)
    >>> features.shape
    torch.Size([10, 5, 256])

    # Normalize the feature vectors:
    >>> features = normalize_features(features)

    >>> loss_function(None, features)
    tensor(4.1987)
    """
    return LOSS(out)


def normalize_features(features):
    """
    >>> seed = torch.manual_seed(0)
    >>> bsz = 10  # Batch size
    >>> n_views = 5  # the number of crops from each image (positive example)
    >>> latent_dim = 256
    >>> features = torch.randn(bsz, n_views, latent_dim)
    >>> features.shape
    torch.Size([10, 5, 256])
    >>> features = normalize_features(features)
    >>> torch.norm(features, dim=2)
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
    """
    # Normalize the feature vectors:
    epsilon = 1e-6
    norms = torch.norm(features, dim=2)[..., None]
    norms += epsilon
    out = features / norms
    return out


def forward_batch(batch, model, normalize=True):
    """
    >>> seed = torch.manual_seed(0)
    >>> pdbpath = 'data/pdb'
    >>> nviews = 2
    >>> batch_size = 3
    >>> out_channels = 256
    >>> num_workers = os.cpu_count()
    >>> dataset = DensityLoader.DensityDataset(pdbpath, nsample=nviews)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=DensityLoader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [len(l) for l in batch]
    [2, 2, 2]
    >>> [[e.shape for e in l] for l in batch]
    [[(29, 35, 27), (23, 21, 36)], [(60, 59, 65), (50, 57, 66)], [(40, 41, 45), (43, 41, 46)]]
    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=out_channels)
    >>> out = forward_batch(batch, model)
    >>> out.shape
    torch.Size([3, 2, 256])
    """
    out = []
    for system in batch:
        out_ = []
        for view in system:
            view = torch.tensor(view[None, None, ...]).float().to(DEVICE)
            # view = torch.autograd.Variable(view, requires_grad=True)
            out_.append(model(view))
        out_ = torch.cat(out_, 0)
        out.append(out_)
    out = torch.stack(out)
    if normalize:
        out = normalize_features(out)
    return out


def batchsizereporter_func(X):
    """
    >>> seed = torch.manual_seed(0)
    >>> pdbpath = 'data/pdb'
    >>> nviews = 2
    >>> batch_size = 3
    >>> out_channels = 256
    >>> num_workers = os.cpu_count()
    >>> dataset = DensityLoader.DensityDataset(pdbpath, nsample=nviews)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=DensityLoader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [[e.shape for e in l] for l in batch]
    [[(29, 35, 27), (23, 21, 36)], [(60, 59, 65), (50, 57, 66)], [(40, 41, 45), (43, 41, 46)]]
    >>> batchsizereporter_func(batch)
    617891
    """
    out = 0
    for li in X:
        for e in li:
            out += e.size
    return out


def train(latent_dim=256,
          pdbpath='data/pdb',
          ext='cif.gz',
          nviews=5,
          batch_size=4,
          n_epochs=10,
          save_each=30,
          print_each=100,
          early_break=np.inf,
          batchsizereporter_func=None,
          batchmemcutoff=np.inf,
          exclude_list=None):
    """
    - nviews: the number of random views for the same system (pdb)

    >>> train(n_epochs=1, print_each=1, batch_size=3, nviews=2, early_break=1, batchsizereporter_func=batchsizereporter_func)
    """
    model = resnet3d.resnet3d(in_channels=1, out_channels=latent_dim)
    dataset = DensityLoader.DensityDataset(pdbpath,
                                           nsample=nviews,
                                           ext=ext,
                                           uniprot_pdb=True,
                                           list_ids_file='training_set.txt.gz',
                                           exclude_list=exclude_list,
                                           skip_error=True)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=DensityLoader.collate_fn,
                                             pin_memory=True,
                                             timeout=20)
    trainer.train(model,
                  loss_function,
                  dataloader,
                  n_epochs,
                  forward_batch,
                  save_each=save_each,
                  print_each=print_each,
                  early_break=early_break,
                  batchsizereporter_func=batchsizereporter_func,
                  batchmemcutoff=batchmemcutoff)


def encode(*args, model):
    """
    Encode the given density map (dmap)
    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=256)
    >>> model = trainer.load_model(model, filename='20221005_model.pt')
    >>> dmap = np.random.uniform(size=(50, 40, 60))
    >>> dmap.shape
    (50, 40, 60)
    >>> v = encode(dmap, model=model)
    >>> v.shape
    (1, 256)

    # encode can accept multiple dmap
    >>> dmap2 = np.random.uniform(size=(60, 50, 40))
    >>> v = encode(dmap, dmap2, model=model)
    >>> v.shape
    (2, 256)
    """
    batch = [[dmap] for dmap in args]
    v = forward_batch(batch, model, normalize=True)
    return v.detach().cpu().numpy()[:, 0, ...]


def encode_pdb(*args, model, sigma=1., spacing=1):
    """
    Encode the given pdb code or pdb file (pdb, mmcif, ...)
    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=256)

    When multiple pdbs are given they are forwarded as a batch (more efficient on gpu, but more memory usage)
    >>> v = encode_pdb('1ycr', '4ci0', model=model)
    >>> v.shape
    (2, 256)
    """
    batch = [[Density(pdb, sigma=sigma, spacing=spacing, padding=(3, 3, 3))[0]] for pdb in args]
    v = forward_batch(batch, model, normalize=True)
    return v.detach().cpu().numpy()[:, 0, ...]


def encode_dir(directory,
               ext,
               model,
               batch_size=1,
               sigma=1.,
               spacing=1.,
               index_dirname='nnindex',
               n_trees=10,
               verbose=True,
               early_break=np.inf):
    """

    Args:
        directory: directory containing structure files
        ext: extension of the structure files to encode
        model: DL model -- encoder
        batch_size: the size of the batch (number of structures forwarded at once)
        sigma: sigma for the synthetic density map
        spacing: spacing for the synthetic density map
        index_dirname: name of the directory to store the index (with nearest neighbor search)
        n_trees: number of trees for the Randomized Partition Trees (RP-Trees) see: annoy (https://github.com/spotify/annoy)
        verbose: print timing
        early_break: take the given first files (for testing)
    Returns:
        nnindex: an NNindex object that can be used to query nearest neighbors (see: class NNindex)

    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=256)
    >>> model = trainer.load_model(model, filename='models/20221005_model.pt')
    >>> nnindex = encode_dir(directory='data/pdb', ext='cif.gz', model=model, early_break=9, index_dirname='nnindex_test', verbose=False)

    The returned index can be used for querying nearest neighbors (see: class NNindex)
    >>> neighbors, distances = nnindex.query(name='3a9r', k=3)
    >>> neighbors
    ['3a9r', '7a9x', '6a9u']
    >>> distances
    [1.0000001192092896, 0.8570012450218201, 0.8555854558944702]

    The index is saved on disk in index_dirname and can be loaded afterward:
    >>> del nnindex
    >>> nnindex = NNindex(256, index_dirname='nnindex_test')
    >>> nnindex.load()
    Loading index with metric: dot
    >>> neighbors, distances = nnindex.query(name='3a9r', k=3)
    >>> neighbors
    ['3a9r', '7a9x', '6a9u']
    >>> distances
    [1.0000001192092896, 0.8570012450218201, 0.8555854558944702]
    """
    dim = model(torch.randn(1, 1, 10, 10, 10).to(DEVICE)).shape[-1]
    nnindex = NNindex(dim, metric='dot', index_dirname=index_dirname)
    dataset = DensityLoader.DensityDataset(directory,
                                           return_name=True,
                                           nsample=1,
                                           ext=ext,
                                           random_chains=False,
                                           random_rotation=False,
                                           sigma=sigma,
                                           skip_error=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=os.cpu_count(),
                                             collate_fn=DensityLoader.collate_fn)
    for i, batch in enumerate(tqdm(dataloader)):
        if i > early_break:
            break
        try:
            densities = [e[0] for e in batch]
            names = [e[1] for e in batch]
            outbatch = encode(*densities, model=model)
            nnindex.add_batch(outbatch, names)
        except RuntimeError:
            print(f'Cannot encode: {names}')
    TIMER.start(f'Building index with {n_trees} trees', verbose=verbose)
    nnindex.build(n_trees=n_trees)
    TIMER.stop()
    return nnindex


def query(model, mrcfilename=None, pdb=None, name=None, vector=None, k=3, search_k=None, index_dirname='nnindex'):
    """

    Args:
        model: DL model -- encoder
        mrcfilename: mrc filename to read the density from
        pdb: pdb code to search neighbors for
        name: name of the vector to search neighbors for (the vector must be in the annoy db)
        vector: vector to search neighbors for
        k: number of neighbors to return
        search_k: number of nodes to inspect during searching (see: https://github.com/spotify/annoy#tradeoffs)

    Returns:
        nnames: the names of the neighbors
        dists: the corresponding distances

    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=256)
    >>> model = trainer.load_model(model, filename='models/20221005_model.pt~20221013-103512~')
    >>> nnames, dists = query(model, name='1ycr', k=3)
    Loading index with metric: dot
    >>> nnames, dists
    (['1ycr', '2zzt', '4q2m'], [1.0000001192092896, 0.9997105002403259, 0.9996662139892578])

    >>> nnames, dists = query(model, mrcfilename='data/1ycr_density.mrc', k=3)
    Loading index with metric: dot
    >>> nnames, dists
    (['6ucw', '4pss', '1s6h'], [0.997984766960144, 0.9968208074569702, 0.9965625405311584])

    """
    dim = model(torch.randn(1, 1, 10, 10, 10).to(DEVICE)).shape[-1]
    nnindex = NNindex(dim, metric='dot', index_dirname=index_dirname)
    nnindex.load()
    if pdb is not None:
        vector = np.squeeze(encode_pdb(pdb, model=model))
    if mrcfilename is not None:
        data, origin, spacing = mrc.mrc_to_array(mrcfilename, normalize=True)
        data, spacing_out = mrc.resample(data, spacing, spacing)
        vector = np.squeeze(encode(data, model=model))
    nnames, dists = nnindex.query(name=name, vector=vector, k=k, search_k=search_k)
    return nnames, dists


def get_similarity(pdb1=None, pdb2=None, dmap1=None, dmap2=None, model=None, sigma=1., spacing=1):
    """
    >>> np.random.seed(0)
    >>> seed = torch.manual_seed(0)
    >>> model = resnet3d.resnet3d(in_channels=1, out_channels=256)
    >>> dmap1 = np.random.uniform(size=(50, 40, 60))
    >>> dmap2 = np.random.uniform(size=(60, 50, 40))
    >>> get_similarity(dmap1=dmap1, dmap2=dmap2, model=model)
    0.9974737
    """
    if pdb1 is not None:
        dmap1 = Density(pdb1, sigma=sigma, spacing=spacing)[0]
    if pdb2 is not None:
        dmap2 = Density(pdb2, sigma=sigma, spacing=spacing)[0]
    v = encode(dmap1, dmap2, model=model)
    sim = v[0].dot(v[1].T)
    return sim


def load_model(modelfilename):
    print(f'Loading DL model: {modelfilename}')
    model = resnet3d.resnet3d(in_channels=1, out_channels=256)
    model = trainer.load_model(model, modelfilename)
    model = model.to(DEVICE)
    return model


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    import logging
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logfilename = 'logs/deminer.log'  # os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--test_dataset', help='Test the full dataset', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')

    parser.add_argument('--query_pdb', help='Query Deminer by pdb id')
    parser.add_argument('--query_mrc', help='Query Deminer by giving an MRC filename')
    parser.add_argument('-k', help='number of nearest neighbors to return', type=int, default=3)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pdbpath', help='Path to the PDB database (default: data/pdb)', default='data/pdb')
    parser.add_argument('--ext',
                        help='Extension of the files to read in the PDB database (default: cif.gz)',
                        default='cif.gz')
    parser.add_argument('--print_each', help='Log printing interval (default: every 100 steps)', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--nviews', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--batchmemcutoff', default=30000000, type=float)
    parser.add_argument('--num_workers', type=int)

    parser.add_argument('--sim', nargs=2, help='Compute the similarity between the given 2 pdb (code or file)')
    parser.add_argument('--model', help='DL-model to load', default='models/20221005_model.pt~20221108-100139~')

    parser.add_argument(
        '--encode_dir',
        help='encode the given directory (see: --ext to select the wanted file extension) and store it in an index')
    args = parser.parse_args()
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()
    exclude_list = np.genfromtxt('data/exclude_list.txt', dtype=str)
    if args.test_dataset:
        dataset = DensityLoader.DensityDataset(pdbpath='data/pdb',
                                               nsample=args.nviews,
                                               ext=args.ext,
                                               uniprot_pdb=False,
                                               list_ids_file=None,
                                               exclude_list=exclude_list,
                                               verbose=True,
                                               skip_error=True)
        if args.num_workers is None:
            num_workers = os.cpu_count()
        else:
            num_workers = args.num_workers
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 collate_fn=DensityLoader.collate_fn,
                                                 pin_memory=True,
                                                 timeout=20)
        for batch in tqdm(dataloader):
            pass
        sys.exit()
    if args.query_pdb is not None or args.query_mrc is not None:
        model = load_model(args.model)
        nnames, dists = query(model, pdb=args.query_pdb, mrcfilename=args.query_mrc, k=args.k)
        for n, d in zip(nnames, dists):
            print(n, d)
    if args.train:
        train(latent_dim=256,
              pdbpath='data/pdb',
              ext=args.ext,
              nviews=args.nviews,
              batch_size=args.batch_size,
              n_epochs=args.n_epochs,
              save_each=30,
              print_each=args.print_each,
              early_break=np.inf,
              batchsizereporter_func=batchsizereporter_func,
              batchmemcutoff=args.batchmemcutoff,
              exclude_list=exclude_list)
    if args.sim is not None:
        if args.model is None:
            print('Please give a DL model using --model')
            sys.exit(1)
        model = load_model(args.model)
        sim = get_similarity(pdb1=args.sim[0], pdb2=args.sim[1], model=model)
        print(sim)

    if args.encode_dir is not None:
        model = load_model(args.model)
        encode_dir(directory=args.encode_dir, ext=args.ext, model=model, batch_size=args.batch_size)
