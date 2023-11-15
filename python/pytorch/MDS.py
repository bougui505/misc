#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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


import gzip
import os

import numpy as np
import scipy.spatial.distance as scidist
import torch

from misc import rec


class Mover(torch.nn.Module):
    """
    >>> mover = Mover(npts=100)
    >>> x = torch.randn(size=(100, 2))
    >>> x = mover(x)
    >>> x.shape
    torch.Size([100, 2])
    """

    def __init__(self, npts, ndims=2):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.randn(size=(npts, ndims)))

    def forward(self, x):
        return x + self.delta


def lossfunc(x, dmat, repulsion=0.0):
    """
    If a dmat value is -1.0 this value is masked in the loss computation
    >>> y = torch.randn(size=(100, 8))
    >>> dmat = torch.cdist(y, y)
    >>> x = torch.randn(size=(100, 2))
    >>> loss1 = lossfunc(x, dmat)
    >>> loss1
    tensor(...)

    Try with a mask
    >>> dmat[0, 1] = -1.0
    >>> dmat[10, 5] = -1.0
    >>> loss2 = lossfunc(x, dmat)
    >>> loss2
    tensor(...)
    >>> loss2 < loss1
    tensor(True)

    Try with repulsion
    >>> loss3 = lossfunc(x, dmat, repulsion=0.01)
    >>> loss3
    tensor(...)
    >>> loss3 > loss2
    tensor(True)
    """
    xmat = torch.cdist(x, x)
    mask = (dmat == -1.0)
    loss_dmat = torch.mean((xmat[~mask] - dmat[~mask])**2)
    if repulsion > 0.0:
        repulsive_mask = (xmat < repulsion)
        loss_rep = torch.mean((xmat[repulsive_mask] - repulsion)**2)
        return loss_dmat + loss_rep
    else:
        return loss_dmat


def pca(dmat, ndims=2):
    """
    >>> x = torch.randn(10, 10)
    >>> dmat = torch.cdist(x, x)
    >>> x = pca(dmat)
    >>> x.shape
    torch.Size([10, 2])
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(dmat - dmat.mean(dim=0))
    eigenvalues = torch.flip(eigenvalues, [0])
    eigenvectors = torch.flip(eigenvectors, [1])
    x = torch.mm(dmat, eigenvectors[:, :ndims])
    return x


def fit(dmat, repulsion, ndims=2, niter=10000, device='cpu', min_delta=1e-6, x=None, return_np=True, verbose=True):
    npts = dmat.shape[0]
    dmat = dmat.to(device)
    if x is None:
        # x = torch.randn((npts, ndims)).to(device)
        x = pca(dmat, ndims=ndims)
    mover = Mover(npts=npts, ndims=ndims).to(device)
    optimizer = torch.optim.Adam(mover.parameters(), amsgrad=False, lr=0.01)
    y = x
    loss = torch.inf
    loss_prev = torch.inf
    for i in range(niter):
        optimizer.zero_grad()
        y = mover(x)
        loss = lossfunc(y, dmat, repulsion=repulsion)
        loss.backward()
        optimizer.step()
        progress = (i+1)/niter
        delta_loss = torch.abs(loss - loss_prev)
        loss_prev = loss
        if verbose:
            print(f'{i=}')
            print(f'{progress=:.2%}')
            print(f'{loss=:.5g}')
            print(f'{delta_loss=:.5g}')
            print('--')
        if delta_loss <= min_delta:
            break
    if return_np:
        return y.detach().cpu().numpy(), loss
    else:
        return y.detach(), loss


def fit_batched(recfile, batch_size, field, nepochs, repulsion=0, ndims=2, niter=10000, device='cpu', min_delta=1e-6, min_delta_epoch=1e-6):
    data, fields = rec.get_data(recfile, selected_fields=None, rmquote=False)
    n = int(max(data['i']))
    p = int(max(data['j']))
    npts = max(n, p) + 1
    print(f"{npts=}")
    x = torch.randn((npts, ndims)).to(device)
    ilist = np.unique(data['i'])
    jlist = np.unique(data['j'])
    loss_prev = torch.inf
    for epoch in range(nepochs):
        batch, batch_inds = subsample(data, batch_size, ilist, jlist)
        dmat = rec_to_mat(field=field, data=batch, fields=fields)
        y, loss = fit(dmat, repulsion=repulsion, ndims=ndims,
                      niter=niter, device=device, min_delta=min_delta, x=x[batch_inds], return_np=False, verbose=False)
        progress = (epoch+1)/nepochs
        delta_loss_epoch = torch.abs(loss - loss_prev)
        loss_prev = loss
        print(f"{epoch=}")
        print(f"{progress=:.2%}")
        print(f"{loss=:.5g}")
        print(f'{delta_loss_epoch=:.5g}')
        print("--")
        x[batch_inds] = torch.clone(y)
        if delta_loss_epoch <= min_delta_epoch:
            break
    return x.detach().cpu().numpy()


def subsample(data, batch_size, ilist, jlist):
    def reindex(ilist):
        mapping = dict()
        ind = 0
        for i in ilist:
            if i not in mapping:
                mapping[i] = ind
                ind += 1
        return mapping
    indset = list(set(ilist) | set(jlist))
    batch_index = np.random.choice(a=indset, size=batch_size, replace=False)
    new_index = reindex(batch_index)
    I = data['i']
    J = data['j']
    sel = np.logical_and(np.isin(I, batch_index), np.isin(J, batch_index))
    out = dict()
    for field in data:
        out[field] = data[field][sel]
    out['i'] = np.asarray([new_index[i] for i in out['i']])
    out['j'] = np.asarray([new_index[j] for j in out['j']])
    return out, batch_index


def rec_to_mat(recfile='', field='', data=[], fields=[]):
    if recfile != '':
        assert len(data) == 0, 'recfile xor data should be given, not both'
        data, fields = rec.get_data(
            recfile, selected_fields=None, rmquote=False)
    if len(data) > 0:
        assert len(fields) > 0, 'If data is given, fields must be given'
    assert field in fields
    assert 'i' in fields, "'i' must be defined in the fields of the recfile to store the row index"
    assert 'j' in fields, "'j' must be defined in the fields of the recfile to store the row index"
    n = int(max(data['i']))
    p = int(max(data['j']))
    maxn = max(n, p) + 1
    dmat = -torch.ones(maxn, maxn)
    # print(f"{dmat.shape=}")
    for index, distance in enumerate(data[field]):
        i = int(data['i'][index])
        j = int(data['j'][index])
        dmat[i, j] = distance
    return dmat


def write_rec_pairwise(recfile, outrecfile, mdsout):
    data, fields = rec.get_data(
        recfile, selected_fields=None, rmquote=False)
    assert 'i' in fields, "'i' must be defined in the fields of the recfile to store the row index"
    assert 'j' in fields, "'j' must be defined in the fields of the recfile to store the row index"
    with gzip.open(outrecfile, 'wt') as gz:
        for index, i in enumerate(data["i"]):
            i = int(i)
            j = int(data["j"][index])
            for field in fields:
                gz.write(f"{field}={data[field][index]}\n")
            gz.write(f"mds_i={list(mdsout[i])}\n")
            gz.write(f"mds_j={list(mdsout[j])}\n")
            mds_dist = np.sqrt(((mdsout[j] - mdsout[i])**2).sum())
            gz.write(f"mds_dist={mds_dist}\n")
            gz.write("--\n")


def write_rec_simple(outrecfile, mdsout):
    with gzip.open(outrecfile, 'wt') as gz:
        for i, mds in enumerate(mdsout):
            mds = list(mds)
            gz.write(f"{i=}\n")
            gz.write(f"{mds=}\n")
            gz.write("--\n")


if __name__ == '__main__':
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        '-r', '--repulsion', help='Repulsion distance (excluded volume)', type=float, default=0.0)
    parser.add_argument(
        '-d', '--dim', help='Dimension of the projection space', type=int, default=2)
    parser.add_argument(
        '--niter', help='Number of fitting iterations', type=int, default=10000)
    parser.add_argument(
        '--min_delta', help="Stop criteria based on min_delta: minimum change in the loss to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement (default=1e-6).", type=float, default=1e-6)
    parser.add_argument(
        '--min_delta_epoch', help="Stop criteria based on min_delta. Same as --min_delta except that the delta is calculated at the end of each epoch. Only used if --batch_size is defined (default=1e-6).", type=float, default=1e-6)
    parser.add_argument(
        '--outfile', help='out filename that stores the output coordinates (gzip format)', default='mds.gz')
    parser.add_argument(
        '--rec', help='Read the distances from the given rec file and the given field name', nargs=2)
    parser.add_argument(
        '--npy', help='Read the distance matrix from the given numpy file (.npy). Must contain a condensed distance matrix as returned bu scipy.spatial.distance.pdist (upper diagonal matrix)')
    parser.add_argument('-bs', '--batch_size',
                        help='Batch size. Default is no batch, fit all at once.', type=int)
    parser.add_argument(
        '--nepochs', help='Number of epochs. Used only if bs is given. Default=10', default=10, type=int)
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument(
        '--test_fit', help='Test fitting random data', action='store_true')
    parser.add_argument(
        '--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{DEVICE=}")

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS |
                            doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit(0)
    if os.path.exists(args.outfile):
        sys.exit(
            f'{args.outfile} exists, remove it or specify a new out filename using --outfile option')
    if args.test_fit:
        y = torch.randn(size=(100, 8))
        dmat = torch.cdist(y, y)
        out, _ = fit(dmat, repulsion=args.repulsion, ndims=args.dim,
                     niter=args.niter, device=DEVICE)
        with gzip.open(args.outfile, 'wt') as gz:
            if out is not None:
                for l in out:
                    gz.write(' '.join([str(e) for e in l]) + "\n")
        sys.exit(0)
    if args.rec is not None:
        recfile, field = args.rec
        basename = os.path.basename(recfile)
        outrecfile_pairwise = "MDS/" + \
            basename.split(".")[0] + "_mds_pairwise.rec.gz"
        outrecfile_simple = "MDS/" + basename.split(".")[0] + "_mds.rec.gz"
        if not os.path.exists("MDS"):
            os.mkdir("MDS")
        assert not os.path.exists(
            outrecfile_pairwise), f"{outrecfile_pairwise} already exists"
        assert not os.path.exists(
            outrecfile_simple), f"{outrecfile_simple} already exists"
        if args.batch_size is None:
            dmat = rec_to_mat(recfile=recfile, field=field)
            out, _ = fit(dmat, repulsion=args.repulsion, ndims=args.dim,
                         niter=args.niter, device=DEVICE, min_delta=args.min_delta)
        else:
            out = fit_batched(recfile=recfile, batch_size=args.batch_size, field=field, nepochs=args.nepochs, repulsion=args.repulsion,
                              ndims=args.dim, niter=args.niter, device=DEVICE, min_delta=args.min_delta, min_delta_epoch=args.min_delta_epoch)
        write_rec_pairwise(
            recfile=recfile, outrecfile=outrecfile_pairwise, mdsout=out)
        write_rec_simple(outrecfile=outrecfile_simple, mdsout=out)
    if args.npy is not None:
        if not os.path.isdir("MDS"):
            os.mkdir("MDS")
        basename = os.path.basename(args.npy)
        outrecfile_simple = "MDS/" + basename.split(".")[0] + "_mds.rec.gz"
        dmat = torch.from_numpy(scidist.squareform(np.load(args.npy)))
        out, _ = fit(dmat, repulsion=args.repulsion, ndims=args.dim,
                     niter=args.niter, device=DEVICE, min_delta=args.min_delta)
        write_rec_simple(outrecfile=outrecfile_simple, mdsout=out)
