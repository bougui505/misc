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

# Create an HDF5 of np arrays with the data organized in batch
import os
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from misc.Timer import Timer


class Index(object):
    """
    >>> if os.path.exists('test.hdf5'):
    ...     os.remove('test.hdf5')
    >>> index = Index('test.hdf5')
    >>> X = np.random.random((10, 128))
    >>> index.add_batch(X)
    >>> X = np.random.random((9, 128))
    >>> index.add_batch(X)
    >>> index.keys()
    ['0', '1']

    # index is an iterator
    >>> for data, labels in index:
    ...     data.shape
    ...     labels
    (10, 128)
    array([b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9'],
          dtype='|S21')
    (9, 128)
    array([b'10', b'11', b'12', b'13', b'14', b'15', b'16', b'17', b'18'],
          dtype='|S21')

    # k-nearest neighbor search
    >>> X = np.random.random((3, 128))
    >>> n_d, n_labels = index.search(X, k=4)
    >>> n_d.shape
    (3, 4)
    >>> n_labels.shape
    (3, 4)

    """
    def __init__(self, h5filename):
        self.h5filename = h5filename
        self.index_builder = 0
        self.batch_index_builder = 0

    def __iter__(self):
        self.batch_index_iter = 0
        return self

    def add_batch(self, batch, labels=None):
        """
        Add a batch of vectors to the index
        """
        n = len(batch)
        if labels is not None:
            assert len(labels) == n
        with h5py.File(self.h5filename, 'a') as h5file:
            grp = h5file.create_group(str(self.batch_index_builder))
            grp.create_dataset('data', data=batch)
            if labels is None:
                labels = np.arange(self.index_builder, self.index_builder + n).astype('S')
            grp.create_dataset('labels', data=labels)
        self.batch_index_builder += 1
        self.index_builder += n

    def keys(self):
        with h5py.File(self.h5filename, 'r') as h5file:
            out = list(h5file.keys())
        return out

    def search(self, X, k):
        """
        k nearest neighbor search
        """
        n = len(X)
        out_distances = np.ones((n, k)) * np.inf
        out_labels = np.zeros((n, k))
        out_labels = out_labels.astype('S')
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        for data, labels in self:
            nn.fit(data)
            d_neighbors, i_neighbors = nn.kneighbors(X)
            labels_neighbors = labels[i_neighbors]
            sel = (d_neighbors < out_distances)
            out_distances[sel] = d_neighbors[sel]
            out_labels[sel] = labels_neighbors[sel]
        return out_distances, out_labels

    def __next__(self):
        if self.batch_index_iter < self.batch_index_builder:
            with h5py.File(self.h5filename, 'r') as h5file:
                data = np.asarray(h5file[str(self.batch_index_iter)]['data'])
                labels = np.asarray(h5file[str(self.batch_index_iter)]['labels'])
            self.batch_index_iter += 1
            return data, labels
        else:
            raise StopIteration


def test(nvecttors=1000000, ntrain=100000, dim=128, nqueries=3, nbatch=10):
    timer = Timer(autoreset=True)

    index = Index(h5filename='test_index.hdf5')
    if os.path.exists('test_index.hdf5'):
        os.remove('test_index.hdf5')

    timer.start('Adding vectors')
    for bid in range(nbatch):
        toadd = nvecttors // nbatch
        X = np.random.random((toadd, dim))
        print(f'{bid}/{nbatch}', X.shape)
        labels = np.asarray([f'{bid}_{i}' for i in range(toadd)]).astype('S')
        index.add_batch(X, labels=labels)
    timer.stop()

    # X = np.random.random((10, dim))
    # labels = np.string_([f'{i}' for i in range(10)])
    # index.read_index('test_som.p')

    i_queries = np.random.choice(len(X), size=nqueries)
    print('Index of queries: ', i_queries)
    X_queries = X[i_queries]
    label_queries = labels[i_queries]
    print('Labels of queries: ', label_queries)

    timer.start('Querying the index')
    d_neighbors, label_neighbors = index.search(X_queries, k=4)
    timer.stop()
    print(label_neighbors)
    print(d_neighbors)


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
