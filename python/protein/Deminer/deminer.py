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
import os
import torch
from misc.pytorch import DensityLoader, contrastive_loss, resnet3d, trainer
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential

LOSS = contrastive_loss.SupConLoss()
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
          batchsizereporter_func=None):
    """
    - nviews: the number of random views for the same system (pdb)

    >>> train(n_epochs=1, print_each=1, batch_size=3, nviews=2, early_break=1, batchsizereporter_func=batchsizereporter_func)
    """
    model = resnet3d.resnet3d(in_channels=1, out_channels=latent_dim)
    dataset = DensityLoader.DensityDataset(pdbpath, nsample=nviews, ext=ext, uniprot_pdb=True)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=DensityLoader.collate_fn)
    trainer.train(model,
                  loss_function,
                  dataloader,
                  n_epochs,
                  forward_batch,
                  save_each=save_each,
                  print_each=print_each,
                  early_break=early_break,
                  batchsizereporter_func=batchsizereporter_func)


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
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pdbpath', help='Path to the PDB database (default: data/pdb)', default='data/pdb')
    parser.add_argument('--ext',
                        help='Extension of the files to read in the PDB database (default: cif.gz)',
                        default='cif.gz')
    parser.add_argument('--print_each', help='Log printing interval (default: every 100 steps)', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--nviews', default=5, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
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
              batchsizereporter_func=batchsizereporter_func)
