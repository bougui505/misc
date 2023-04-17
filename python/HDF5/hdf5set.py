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
import os
import h5py
import numpy as np
from tqdm import tqdm
from misc.Timer import Timer
import random
import string


class HDF5set(object):
    """
    Instanciate the object
    >>> h5filename = '/tmp/test.h5'
    >>> if os.path.exists(h5filename):
    ...     os.remove(h5filename)
    >>> hdf5set = HDF5set(h5filename)
    # h5py version: 2.10.0

    List keys in hdf5set (returns a set) => should be an empty set
    >>> hdf5set.keys()
    set()

    Adding a single data point
    >>> key = 'a'
    >>> data = np.random.uniform(size=(10, 10))
    >>> hdf5set.add(key, data)

    Adding a batch
    >>> batch = np.random.uniform(size=(32, 10, 10))
    >>> keys = [f'{e}' for e in range(32)]
    >>> hdf5set.add_batch(keys, batch)

    Retrieve a single data point
    >>> data = hdf5set.get('3')
    >>> data.shape
    (10, 10)
    >>> data
    array([[...

    Retrieve a batch
    >>> batch = hdf5set.get_batch(['3', '6', '8'])
    >>> batch.shape
    (3, 10, 10)
    >>> batch
    array([[[...

    Test the exception when the same key is given when adding data
    >>> hdf5set.add('a', data)
    # key "a" already exists in /tmp/test.h5

    Creating datasets in subgroups. 'ac' group contains group 'a' and a dataset
    >>> hdf5set.add('ab', np.random.uniform(size=(10, 10)))
    >>> hdf5set.add('aca', np.random.uniform(size=(10, 10)))
    >>> hdf5set.add('ac', np.random.uniform(size=(10, 10)))
    >>> hdf5set.get('ac')
    array([[...

    List keys in hdf5set (returns a set)
    >>> sorted(hdf5set.keys())
    ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6', '7', '8', '9', 'a', 'ab', 'ac', 'aca']


    >>> os.remove(h5filename)
    """
    def __init__(self, h5filename, mode='a'):
        """
        """
        print(f'# h5py version: {h5py.__version__}')
        self.h5filename = h5filename
        self.h5file = h5py.File(self.h5filename, mode)

    def keys(self):
        if not os.path.exists(self.h5filename):
            return set()
        return set(self.h5file.keys())

    def add_batch(self, keys, batch):
        """
        """
        for i, key in enumerate(keys):
            data = batch[i]
            try:
                self.h5file.create_dataset(name=key, data=data)
            except RuntimeError:
                print(f'# key "{key}" already exists in {self.h5filename}')
            except ValueError:
                print(f'# key "{key}" already exists in {self.h5filename}')

    def add(self, key, data):
        """
        """
        keys = [
            key,
        ]
        batch = data[None, ...]
        self.add_batch(keys, batch)

    def get(self, key):
        """
        """
        data = self.h5file[key][()]
        return data

    def get_batch(self, keys):
        batch = []
        for key in keys:
            data = self.h5file[key][()]
            batch.append(data)
        batch = np.asarray(batch)
        return batch


def random_key(min_len=8, max_len=128):
    """
    >>> np.random.seed(0)
    >>> random_key()
    'VaddNjtvYKxgyymbMNxUyrLznijuZqZfpVasJyXZDttoNGbjGFkx'
    """
    klen = np.random.randint(low=min_len, high=max_len)
    key = np.random.choice(list(string.ascii_letters), size=klen)
    key = ''.join(key)
    return key


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
    parser.add_argument(
        '--test_long',
        help=
        'Test the code by creating an hdf5 file. Take n arguments: the number of elements to store and the shape of the element',
        type=int,
        nargs='+')
    parser.add_argument('--speed_test_read', help='Speed test for reading the given hdf5 file')
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
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.test_long is not None:
        timer = Timer(autoreset=True, colors=True)
        h5filename = 'test.h5'
        if os.path.exists(h5filename):
            os.remove(h5filename)
        hdf5set = HDF5set(h5filename)
        n = args.test_long[0]
        s = tuple(args.test_long[1:])
        print()
        timer.start(message=f'# writing {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = random_key()
            v = np.random.uniform(size=s)
            hdf5set.add(k, v)
        timer.stop()
        hdf5set = HDF5set(h5filename, mode='r')
        timer.start(message='# reading keys')
        keys = list(hdf5set.keys())
        timer.stop()
        random.shuffle(keys)
        timer.start(message=f'# reading {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = keys[i]
            v = hdf5set.get(k)
        timer.stop()
        sys.exit()

    if args.speed_test_read is not None:
        timer = Timer(autoreset=True, colors=True)
        h5filename = args.speed_test_read
        hdf5set = HDF5set(h5filename, mode='r')
        print()
        timer.start(message='# reading keys')
        keys = hdf5set.keys()
        timer.stop()
        timer.start(message='# reading data ...')
        keys = list(keys)
        random.shuffle(keys)
        for k in tqdm(keys):
            v = hdf5set.get(k)
        timer.stop()
        sys.exit()
