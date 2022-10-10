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
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
import numpy as np
from misc.Timer import Timer
from tqdm import tqdm
import scipy.spatial.distance as scidist

TIMER = Timer(autoreset=True)


def test(npts=1000,
         dim=512,
         metric="euclidean",
         ntrees=10,
         indexfilename='test.ann',
         nneighbors=3,
         compare_with_exact_nn=True):
    """
    metric: "angular", "euclidean", "manhattan", "hamming", "dot"
    """
    print("Testing with following arguments:")
    print(locals())
    if compare_with_exact_nn:
        X = []
    else:
        X = None
    TIMER.start("Building index")
    index = AnnoyIndex(dim, metric)
    for i in tqdm(range(npts)):
        mu = np.random.choice(100)
        v = np.random.normal(loc=mu, size=dim)
        index.add_item(i, v)
        if X is not None:
            X.append(v)
    index.build(ntrees)
    TIMER.stop()
    if X is not None:
        X = np.asarray(X)
    TIMER.start("Saving index")
    index.save(indexfilename)
    TIMER.stop()
    del index
    TIMER.start("Loading index")
    index = AnnoyIndex(dim, metric)
    index.load(indexfilename)
    TIMER.stop()
    query = np.random.normal(size=(dim))
    TIMER.start(f"Querying {nneighbors} nearest neighbors")
    ann = index.get_nns_by_vector(query, nneighbors, search_k=-1)  # Approximate nearest neighbors
    TIMER.stop()
    print(ann)
    if compare_with_exact_nn:
        TIMER.start("Exact nearest neighbors")
        nbrs = NearestNeighbors(n_neighbors=nneighbors, algorithm='auto').fit(X)
        distance, enn = nbrs.kneighbors(query[None, ...])  # Exact nearest neighbors
        TIMER.stop()
        enn = enn[0]
        print(enn)
        jaccard = len(set(ann) & set(enn)) / len(set(ann) | set(enn))
        print('Jaccard coefficient:', jaccard)
    TIMER.start("cdist")
    dmat = scidist.cdist(query[None, ...], X)
    print(dmat.argsort(axis=1)[:, :nneighbors][0])
    TIMER.stop()


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
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    parser.add_argument('-n', '--npts', help='Number of points (1000)', default=1000, type=int)
    parser.add_argument('-d', '--dim', help='Space dimension (512)', default=512, type=int)
    parser.add_argument('--metric',
                        help='Metric to use ("angular", "euclidean", "manhattan", "hamming", "dot"); ("euclidean")',
                        default='euclidean')
    parser.add_argument('--ntrees', help='Number of random trees (10)', default=10, type=int)
    parser.add_argument('--indexfilename',
                        help='Approximate Nearest Neighbors index filename (test.ann)',
                        default='test.ann')
    parser.add_argument('--nn', help='Number of nearest neighbors to search for (3)', default=3, type=int)
    parser.add_argument('--no_enn',
                        help='Do not compare results with exact neighbors search',
                        action='store_false',
                        dest='enn')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
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

    test(npts=args.npts,
         dim=args.dim,
         metric=args.metric,
         ntrees=args.ntrees,
         indexfilename=args.indexfilename,
         nneighbors=args.nn,
         compare_with_exact_nn=args.enn)
