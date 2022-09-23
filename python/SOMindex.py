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
from quicksom.som import SOM, ArrayDataset
from misc.Timer import Timer
import numpy as np
import torch
import h5py
import scipy.spatial.distance as scidist

TIMER = Timer(autoreset=True)


class SOMindex(object):
    def __init__(self,
                 dim,
                 m=50,
                 n=50,
                 alpha=None,
                 sigma=None,
                 periodic=True,
                 device='cpu',
                 somfilename='som.p',
                 h5filename='index.hdf5'):
        self.device = device
        self.somfilename = somfilename
        self.h5filename = h5filename
        self.som = SOM(m=m, n=n, dim=dim, alpha=alpha, sigma=sigma, periodic=periodic, device=device)

    def read_index(self, somfile):
        self.som.load_pickle(somfile)

    def train(self, Xt):
        """
        Train the SOM using Xt
        """
        Xt = torch.tensor(Xt, device=self.device)
        self.som.fit(dataset=Xt)
        self.som.save_pickle(self.somfilename)

    def add(self, X, labels=None):
        """
        Add a batch of vectors X to the index
        """
        X = torch.tensor(X, device=self.device).float()
        dataset = ArrayDataset(X, labels=labels)
        bmus, errors, labels = self.som.predict(dataset)
        labels = np.string_(labels)
        self.__add_to_hdf5__(X, bmus, labels)

    def __add_to_hdf5__(self, X, bmus, labels):
        bmus = np.ravel_multi_index(bmus.T, (self.som.m, self.som.n))
        with h5py.File(self.h5filename, 'a') as h5file:
            for bmu in np.unique(bmus):
                sel = (bmus == bmu)
                grp = h5file.require_group(str(bmu))
                for v, label in zip(X[sel], labels[sel]):
                    grp.create_dataset(label, data=v)

    def search(self, X, k):
        """
        Search the index for the k nearest neighbors
        """
        X = torch.tensor(X).float()
        bmus, errors = self.som.inference_call(X)
        bmus = torch.squeeze(bmus)
        bmus = np.asarray([int(e) for e in bmus])
        search_results = self.__search_in_hdf5__(X, bmus, k)
        print(search_results)

    def __search_in_hdf5__(self, X, bmus, k):
        with h5py.File(self.h5filename, 'r') as h5file:
            results = []
            # for bmu in np.unique(bmus):
            for i, bmu in enumerate(bmus):
                grp = h5file[str(bmu)]
                labels = np.asarray(list(grp.keys()))
                members = []
                for label in labels:
                    members.append(grp[label])
                members = np.asarray(members)
                # sel = (bmus == bmu)
                queries = X[i][None, ...]
                cdist = scidist.cdist(queries, members)
                ids = cdist.argsort(axis=1)
                search_result = np.squeeze(labels[ids])[:k]
                results.append(search_result)
        results = np.asarray(results)
        return results


def test(nvecttors=1000000, ntrain=10000, dim=128, nqueries=3, nbatch=10):
    timer = Timer(autoreset=True)

    # timer.start('Random training data generation')
    # Xt = np.random.random((ntrain, dim))
    # timer.stop()

    index = SOMindex(dim, m=10, n=10, somfilename='test_som.p', h5filename='test_index.hdf5')
    # if os.path.exists('test_index.hdf5'):
    #     os.remove('test_index.hdf5')

    # timer.start('SOM training')
    # index.train(Xt)
    # timer.stop()

    # timer.start('Adding vectors')
    # for bid in range(nbatch):
    #     toadd = nvecttors // nbatch
    #     X = np.random.random((toadd, dim))
    #     print(f'{bid}/{nbatch}', X.shape)
    #     labels = np.asarray([f'{bid}_{i}' for i in range(toadd)])
    #     index.add(X, labels=labels)
    # timer.stop()

    X = np.random.random((10, dim))
    labels = np.string_([f'{i}' for i in range(10)])
    index.read_index('test_som.p')

    i_queries = np.random.choice(len(X), size=nqueries)
    print('Index of queries: ', i_queries)
    X_queries = X[i_queries]
    label_queries = labels[i_queries]
    print('Labels of queries: ', label_queries)

    timer.start('Querying the index')
    index.search(X_queries, k=4)
    timer.stop()


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
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        test()
        sys.exit()
