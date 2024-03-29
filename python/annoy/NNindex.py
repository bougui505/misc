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
"""
Helper class to generate an Annoy index for fast Approximate Nearest neighbors search.
See: https://github.com/spotify/annoy

>>> np.random.seed(0)
>>> nnindex = NNindex(128)
>>> for name in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
...     nnindex.add(np.random.normal(size=(128)), name)
>>> nnindex.build(10)
>>> nnindex.query('c', k=3)
(['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])
"""

import os
from annoy import AnnoyIndex
import h5py
import numpy as np
from functools import partial


def autohash(inp, maxdepth=np.inf):
    """

    Args:
        inp:
        maxdepth:

    Returns:

    >>> autohash('abc')
    'a/b/c/'
    >>> autohash(0)
    '0/'
    >>> autohash(1234)
    '1/2/3/4/'
    >>> autohash(1234, maxdepth=2)
    '1/2/34/'
    """
    inp = str(inp)
    out = ''
    for i, char in enumerate(inp):
        if i < maxdepth:
            out += char + '/'
        else:
            out += inp[i:] + '/'
            break
    return out


class NNindex(object):
    """

    Attributes:
        mapping:
        mappingfilename:
        i:
        is_on_disk:
        index_dirname:
        annoyfilename:
        index:
        metric:
        dim:

    Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"

    >>> np.random.seed(0)
    >>> nnindex = NNindex(128)
    >>> for name in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    ...     nnindex.add(np.random.normal(size=(128)), name)
    >>> nnindex.build(10)
    >>> nnindex.query('c', k=3)
    (['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])

    Querying a vector
    >>> nnindex.query(vector=np.random.normal(size=(128)), k=3)
    (['g', 'c', 'f'], [13.61835765838623, 15.400964736938477, 15.585434913635254])

    The search_k parameter can be set
    >>> nnindex.query('c', k=3, search_k=100)
    (['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])

    Try loading:
    >>> del nnindex
    >>> nnindex = NNindex(128)
    >>> nnindex.annoyfilename
    'nnindex/annoy.ann'
    >>> nnindex.load()
    Loading index with metric: euclidean
    >>> nnindex.query('c', k=3)
    (['c', 'g', 'e'], [0.0, 14.265301704406738, 14.367243766784668])

    Adding batch
    >>> nnindex = NNindex(128)
    >>> batch = np.random.normal(size=(4, 128))
    >>> names = ['w', 'x', 'y', 'z']
    >>> nnindex.add_batch(batch, names)

    Hash function testing
    >>> del nnindex
    >>> nnindex = NNindex(128)
    >>> for name in ['abc', 'bcd', 'cde', 'ded', 'efg', 'fgh', 'ghi']:
    ...     nnindex.add(np.random.normal(size=(128)), name)
    >>> nnindex.build(10)
    >>> nnindex.query('ghi', k=3)
    (['ghi', 'cde', 'fgh'], [0.0, 13.730531692504883, 14.671173095703125])

    Same names is not a problem
    >>> nnindex = NNindex(128)
    >>> nnindex.add(np.random.normal(size=(128)), 'a')
    >>> nnindex.add(np.random.normal(size=(128)), 'b')
    >>> nnindex.add(np.random.normal(size=(128)), 'a')
    >>> nnindex.build(10)
    >>> nnindex.query('a', k=3)
    (['a', 'b', 'a'], [0.0, 15.647073745727539, 15.918701171875])

    """

    def __init__(self,
                 dim,
                 metric='euclidean',
                 index_dirname='nnindex',
                 hash_func_index=autohash,
                 hash_func_name=autohash):
        """

        Args:
            dim:
            metric: any of "angular", "euclidean", "manhattan", "hamming", or "dot"
            index_dirname:
            hash_func_index:
            hash_func_name:
        """
        self.dim = dim
        self.index_dirname = index_dirname
        if not os.path.isdir(index_dirname):
            os.makedirs(index_dirname)
        self.annoyfilename = f'{index_dirname}/annoy.ann'
        self.mappingfilename = f'{index_dirname}/mapping.h5'
        self.index = AnnoyIndex(dim, metric)
        self.is_on_disk = False
        self.mapping = Mapping(self.mappingfilename, hash_func_index=hash_func_index, hash_func_name=hash_func_name)
        self.metric = metric
        if not self.mapping.h5f.attrs.__contains__('metric'):
            # Store the metric in the h5f
            self.mapping.h5f.attrs['metric'] = metric
        self.i = 0

    def add(self, v, name):
        """

        Args:
            v:
            name:

        """
        if not self.is_on_disk:
            self.index.on_disk_build(self.annoyfilename)
            self.is_on_disk = True
        self.index.add_item(self.i, v)
        self.mapping.add(self.i, name)
        self.i += 1

    def add_batch(self, batch, names):
        """

        Args:
            batch:
            names:

        """
        for v, name in zip(batch, names):
            self.add(v, name)

    def build(self, n_trees):
        """

        Args:
            n_trees:

        """
        self.index.build(n_trees)

    def query(self, name=None, vector=None, k=1, search_k=None):
        """

        Args:
            name: name of the vector for querying neighbors. The name corresponds to a vector in the annoy DB
            vector: vector to search neigbors of. Vector or name must be defined not both
            k: number of neighbors to retrieve
            search_k: number of nodes to inspect during searching (see: https://github.com/spotify/annoy#tradeoffs)

        Returns:
            nnames: the names of the neighbors
            dists: the corresponding distances

        """
        if name is not None:
            inp = self.mapping.name_to_index(name)
            nnfunc = partial(self.index.get_nns_by_item, n=k, include_distances=True)
        if vector is not None:
            inp = vector
            nnfunc = partial(self.index.get_nns_by_vector, n=k, include_distances=True)
        if search_k is not None:
            nnfunc = partial(nnfunc, search_k=search_k)
        knn, dists = nnfunc(inp)
        nnames = []
        for i in knn:
            nnames.append(self.mapping.index_to_name(i))
        return nnames, dists

    def load(self):
        """

        """
        self.metric = self.mapping.h5f.attrs['metric']
        print(f'Loading index with metric: {self.metric}')
        self.index = AnnoyIndex(self.dim, self.metric)
        self.index.load(self.annoyfilename)


class Mapping(object):
    """

    Attributes:
        h5fname:
        verbose:
        h5f:
        hash_func_index:
        hash_func_name:

    >>> mapping = Mapping('test.h5')
    >>> mapping.add(0, 'toto')

    >>> mapping.index_to_name(0)
    'toto'
    >>> mapping.name_to_index('toto')
    0
    """

    def __init__(self, h5fname, hash_func_index=lambda x: x, hash_func_name=lambda x: x, verbose=False):
        """

        Args:
            h5fname:
            hash_func_index:
            hash_func_name:
            verbose:

        """
        self.verbose = verbose
        self.hash_func_index = hash_func_index
        self.hash_func_name = hash_func_name
        self.h5fname = h5fname
        self.h5f = h5py.File(self.h5fname, 'a')
        self.h5f.require_group('name_to_index')
        self.h5f.require_group('index_to_name')

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tra):
        if typ is None:
            self.__del__()

    def __del__(self):
        self.h5f.close()
        if self.verbose:
            print(f'{self.h5fname} closed')

    def add(self, number, name):
        """

        Args:
            number:
            name:

        """
        number_hash = self.hash_func_index(number)
        group = self.h5f['index_to_name']
        leaf = group.require_group(str(number_hash))
        leaf.attrs[str(number)] = name

        name_hash = self.hash_func_name(name)
        group = self.h5f['name_to_index']
        leaf = group.require_group(name_hash)
        leaf.attrs[name] = number

    def index_to_name(self, number):
        """

        Args:
            number:

        Returns:

        """
        number_hash = self.hash_func_index(number)
        group = self.h5f['index_to_name'][str(number_hash)]
        return group.attrs[str(number)]

    def name_to_index(self, name):
        """

        Args:
            name:

        Returns:

        """
        name_hash = self.hash_func_name(name)
        return self.h5f['name_to_index'][name_hash].attrs[str(name)]


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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
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
