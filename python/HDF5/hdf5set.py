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
    # 'HDF5set' opened. Must be closed using 'HDF5set.close()' after usage

    List keys in hdf5set (returns a set) => should be an empty set
    >>> hdf5set.get_keys()
    {'flat'}

    Check if is flat
    >>> hdf5set.flat
    True

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
    >>> sorted(hdf5set.get_keys())
    ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '4', '5', '6', '7', '8', '9', 'a', 'ab', 'ac', 'aca', 'flat']
    >>> hdf5set.close()


    >>> os.remove(h5filename)

    Try for a non-flat hdf5set
    >>> hdf5set = HDF5set(h5filename, flat=False)
    # h5py version: 2.10.0
    # 'HDF5set' opened. Must be closed using 'HDF5set.close()' after usage
    >>> hdf5set.flat
    False
    >>> hdf5set.close()
    # writing keys in /tmp/test.h5
    # writing groups in /tmp/test.h5

    When reloading the file
    >>> hdf5set = HDF5set(h5filename, flat=False, chunk_size=2)
    # h5py version: 2.10.0
    # 'HDF5set' opened. Must be closed using 'HDF5set.close()' after usage
    # reloading /tmp/test.h5 file, reading 'flat' field
    # reloading /tmp/test.h5 file, reading 'keys' field
    # reloading /tmp/test.h5 file, reading 'groups' field
    >>> hdf5set.flat
    False
    >>> hdf5set.add('a', np.random.uniform(size=(10, 10)))
    # creating group 0
    >>> hdf5set.keys
    {'a': '0/a'}
    >>> hdf5set.add('b', np.random.uniform(size=(10, 10)))
    >>> hdf5set.keys
    {'a': '0/a', 'b': '0/b'}
    >>> hdf5set.add('b', np.random.uniform(size=(10, 10)))
    # key "b" already exists in /tmp/test.h5
    >>> hdf5set.add('c', np.random.uniform(size=(10, 10)))
    # creating group 1
    >>> hdf5set.keys
    {'a': '0/a', 'b': '0/b', 'c': '1/c'}
    >>> sorted(hdf5set.get_keys())
    ['a', 'b', 'c']
    >>> hdf5set.groups
    {0: 2, 1: 1}
    >>> hdf5set.close()
    # writing keys in /tmp/test.h5
    # writing groups in /tmp/test.h5

    Reloading the file
    >>> hdf5set = HDF5set(h5filename, flat=False, chunk_size=2)
    # h5py version: 2.10.0
    # 'HDF5set' opened. Must be closed using 'HDF5set.close()' after usage
    # reloading /tmp/test.h5 file, reading 'flat' field
    # reloading /tmp/test.h5 file, reading 'keys' field
    # reloading /tmp/test.h5 file, reading 'groups' field
    >>> hdf5set.keys
    {'a': '0/a', 'b': '0/b', 'c': '1/c'}
    >>> sorted(hdf5set.get_keys())
    ['a', 'b', 'c']
    >>> hdf5set.groups
    {'0': '2', '1': '1'}
    >>> hdf5set.get('a')
    array([[...
    >>> batch = hdf5set.get_batch(['a', 'c'])
    >>> hdf5set.close()
    # writing keys in /tmp/test.h5
    # writing groups in /tmp/test.h5
    """
    def __init__(self, h5filename, mode='a', flat=True, chunk_size=1024):
        """
        - flat: if True store all dataset in '/' group
                else create group chunks of size chunk_size
        """
        print(f'# h5py version: {h5py.__version__}')
        print(
            f"# '{self.__class__.__name__}' opened. Must be closed using '{self.__class__.__name__}.close()' after usage"
        )
        self.h5filename = h5filename
        self.h5file = h5py.File(self.h5filename, mode)
        self.mode = mode
        try:
            self.h5file.create_dataset('flat', data=flat)
        except (RuntimeError, ValueError):
            print(f"# reloading {h5filename} file, reading 'flat' field")
        self.flat = self.is_flat()
        self.chunk_size = chunk_size
        if not self.flat:
            if 'keys' not in self.h5file:
                self.h5file.create_dataset('keys', data=list())
            else:
                print(f"# reloading {h5filename} file, reading 'keys' field")
            self.keys = dict(self.h5file['keys'][()].astype('U'))
            if 'groups' not in self.h5file:
                self.h5file.create_dataset('groups', data=list())
            else:
                print(f"# reloading {h5filename} file, reading 'groups' field")
            self.groups = dict(self.h5file['groups'][()].astype('U'))

    def is_flat(self):
        return self.h5file['flat'][()]

    def get_keys(self):
        if self.flat:
            return set(self.h5file.keys())
        else:
            return set(self.keys.keys())

    @property
    def current_group(self):
        group_ids = self.groups.keys()
        if len(group_ids) > 0:
            group_current = max(group_ids)
            size_current = self.groups[group_current]
        else:
            group_current = -1
            size_current = 0
        if size_current >= self.chunk_size or len(group_ids) == 0:
            group_current += 1
            print(f"# creating group {group_current}")
            self.h5file.create_group(str(group_current))
            self.groups[group_current] = 0
        return str(group_current)

    def add_batch(self, keys, batch):
        """
        """
        for i, key in enumerate(keys):
            data = batch[i]
            if self.flat:
                if key not in self.h5file:
                    self.h5file.create_dataset(name=key, data=data)
                else:
                    print(f'# key "{key}" already exists in {self.h5filename}')
            else:
                if key not in self.keys:
                    group_id = self.current_group
                    self.keys[key] = f"{group_id}/{key}"
                    self.groups[int(group_id)] += 1
                    self.h5file[group_id].create_dataset(name=key, data=data)
                else:
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
        if not self.flat:
            key = self.keys[key]
        data = self.h5file[key][()]
        return data

    def get_batch(self, keys):
        batch = []
        for key in keys:
            data = self.get(key)
            batch.append(data)
        batch = np.asarray(batch)
        return batch

    def close(self):
        if not self.flat:
            if self.mode == 'w' or self.mode == 'a':
                print(f"# writing keys in {self.h5filename}")
                self.replace_metadata('keys', self.keys)
                print(f"# writing groups in {self.h5filename}")
                self.replace_metadata('groups', self.groups)
        self.h5file.close()

    def replace_metadata(self, name, data):
        del self.h5file[name]
        data = list(data.items())
        self.h5file.create_dataset(name=name, data=np.asarray(data, dtype='S'))


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
    parser.add_argument('--chunk',
                        help='Do not use a flat architecture. Create chunk of the given chunk size (e.g. 1024)',
                        type=int)
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument(
        '--test_long',
        help=
        'Test the code by creating an hdf5 file. Take 2 arguments: the number of elements to store and the size of the element',
        type=int,
        nargs=2)
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

    if args.chunk is not None:
        flat = False
        chunk_size = args.chunk
    else:
        flat = True
        chunk_size = None
    if args.test_long is not None:
        timer = Timer(autoreset=True, colors=True)
        h5filename = 'test.h5'
        if os.path.exists(h5filename):
            os.remove(h5filename)
        hdf5set = HDF5set(h5filename, flat=flat, chunk_size=chunk_size)
        n = args.test_long[0]
        s = args.test_long[1]
        print()
        timer.start(message=f'# writing {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = random_key()
            v = np.random.uniform(size=s)
            hdf5set.add(k, v)
        timer.stop()
        hdf5set.close()
        hdf5set = HDF5set(h5filename, mode='r', flat=flat, chunk_size=chunk_size)
        timer.start(message='# reading keys')
        keys = list(hdf5set.get_keys())
        timer.stop()
        random.shuffle(keys)
        timer.start(message=f'# reading {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = keys[i]
            v = hdf5set.get(k)
        timer.stop()
        hdf5set.close()
        sys.exit()

    if args.speed_test_read is not None:
        timer = Timer(autoreset=True, colors=True)
        h5filename = args.speed_test_read
        hdf5set = HDF5set(h5filename, mode='r', flat=flat, chunk_size=chunk_size)
        print()
        timer.start(message='# reading keys')
        keys = hdf5set.get_keys()
        timer.stop()
        timer.start(message='# reading data ...')
        keys = list(keys)
        random.shuffle(keys)
        for k in tqdm(keys):
            v = hdf5set.get(k)
        timer.stop()
        sys.exit()
