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

# General Multidimensional scaling from points with generic distance function

import gzip
import os

import numpy as np
import torch
from misc.pytorch.MDS import fit

from misc import rec


class ExploredRatio(object):
    def __init__(self, npts):
        self.visited = set()
        self.npts = npts
        self.ratio_prev = 0.0

    def update(self, inds):
        self.visited.update(set(inds))
        ratio = self.ratio_prev + len(self.visited) / self.npts
        # print(ratio, self.ratio_prev)
        if ratio % 1 == 0.0:
            self.ratio_prev += 1.0
            self.visited = set()
        return ratio


class Sampler(object):
    def __init__(self, npts, batchsize, data) -> None:
        self.npts = npts
        self.pool = np.arange(self.npts)
        self.batchsize = batchsize
        self.data = data

    def get(self):
        if self.batchsize > len(self.pool):
            n = len(self.pool)
        else:
            n = self.batchsize
        inds = np.random.choice(
            a=self.pool, size=n, replace=False)
        self.pool = np.asarray(list(set(self.pool) - set(inds)))
        if len(self.pool) == 0:
            self.pool = np.arange(self.npts)
        out = dict()
        for field in self.data:
            out[field] = self.data[field][inds]
        return out, inds


def fit_points(recfile, batchsize, nepochs, repulsion, ndims, niter, device, min_delta, min_delta_epoch, checkpoint_file):
    data, fields = rec.get_data(
        recfile, selected_fields=None, rmquote=False)
    print(f"{fields=}")
    npts = len(data[list(data.keys())[0]])
    print(f"{npts=}")
    x = torch.randn((npts, ndims)).to(device)
    loss_prev = torch.inf
    explored = ExploredRatio(npts)
    sampler = Sampler(npts=npts, batchsize=batchsize, data=data)
    epoch = 0
    while epoch < nepochs:
        batch, inds = sampler.get()
        dmat = torch.from_numpy(distance_function(batch))
        y, loss = fit(dmat, repulsion, ndims=ndims, niter=niter, device=device,
                      min_delta=min_delta, x=x[inds], return_np=False, verbose=False)
        delta_loss_epoch = torch.abs(loss - loss_prev)
        loss_prev = loss
        explored_ratio = explored.update(inds)
        epoch = int(explored_ratio)
        progress = (epoch+1)/nepochs
        print(f"{epoch=}")
        print(f"{progress=:.2%}")
        print(f"{loss=:.5g}")
        print(f'{delta_loss_epoch=:.5g}')
        print(f'{explored_ratio=:.2%}')
        x[inds] = torch.clone(y)
        if explored_ratio % 1 == 0:
            print(f"{checkpoint_file=}")
            torch.save(x, checkpoint_file)
        print("--")
        if delta_loss_epoch <= min_delta_epoch:
            break
    torch.save(x, checkpoint_file)
    return x.detach().cpu().numpy()


def write_rec(recfile, outrecfile, mdsout):
    data, fields = rec.get_data(
        recfile, selected_fields=None, rmquote=False)
    npts = len(data[list(data.keys())[0]])
    with gzip.open(outrecfile, 'wt') as gz:
        for i in range(npts):
            for field in fields:
                gz.write(f"{field}={data[field][i]}\n")
            gz.write(f"mds={list(mdsout[i])}\n")
            gz.write("--\n")


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
        '--rec', help='Read the data from the given rec file')
    parser.add_argument('-bs', '--batch_size',
                        help='Batch size', type=int, default=100)
    parser.add_argument(
        '--distance', help='Script containing the distance function that will be imported to the code and name of the function. The function takes data as argument (dictionnary extracted from the rec file) and returns a square distance matrix', nargs=2, metavar=['file.py', 'funcname'])
    parser.add_argument(
        '--niter', help='Number of fitting iterations', type=int, default=10000)
    parser.add_argument(
        '--nepochs', help='Number of epochs. Default=10', default=10, type=int)
    parser.add_argument(
        '-r', '--repulsion', help='Repulsion distance (excluded volume)', type=float, default=0.0)
    parser.add_argument(
        '-d', '--dim', help='Dimension of the projection space', type=int, default=2)
    parser.add_argument(
        '--min_delta', help="Stop criteria based on min_delta: minimum change in the loss to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement (default=1e-6).", type=float, default=1e-6)
    parser.add_argument(
        '--min_delta_epoch', help="Stop criteria based on min_delta. Same as --min_delta except that the delta is calculated at the end of each epoch. Only used if --batch_size is defined (default=1e-6).", type=float, default=1e-6)
    parser.add_argument('--outbasename', help='Basename for the output files')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument(
        '--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

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
        sys.exit()
    if args.rec is not None:
        if args.outbasename is None:
            sys.exit('Please provide --outbasename')
        if args.distance is not None:
            modulename = os.path.splitext(args.distance[0])[0]
            func = args.distance[1]
            module = __import__(
                modulename, globals(), locals())
            distance_function = getattr(module, func)
            print(f"{distance_function=}")
        out = fit_points(recfile=args.rec, batchsize=args.batch_size, nepochs=args.nepochs,
                         repulsion=args.repulsion, ndims=args.dim, niter=args.niter, device=DEVICE, min_delta=args.min_delta, min_delta_epoch=args.min_delta_epoch, checkpoint_file=args.outbasename+".pt")
        write_rec(recfile=args.rec,
                  outrecfile=args.outbasename+".rec.gz", mdsout=out)
