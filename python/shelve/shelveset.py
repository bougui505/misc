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
import shelve
import numpy as np
import string
from misc.Timer import Timer
from tqdm import tqdm
import random


class Shelveset(object):
    """
    >>> filename = '/tmp/test.shelve'
    >>> if os.path.exists(filename):
    ...     os.remove(filename)
    >>> shelveset = Shelveset(filename)
    >>> shelveset.add('a', np.random.uniform(size=(4, 4)))
    >>> shelveset.get('a').shape
    (4, 4)
    >>> shelveset.add_batch(['b', 'c', 'd'], np.random.uniform(size=(3, 4, 4)))
    >>> shelveset.get_batch(['b', 'd']).shape
    (2, 4, 4)
    """
    def __init__(self, filename):
        self.shelve = shelve.open(filename)

    def get_keys(self):
        return self.shelve.keys()

    def add(self, key, data):
        if key not in self.shelve:
            self.shelve[key] = data
        else:
            print(f'# key "{key}" already exists in {self.filename}')

    def add_batch(self, keys, batch):
        for i, key in enumerate(keys):
            data = batch[i]
            self.add(key, data)

    def get(self, key):
        data = self.shelve[key]
        return data

    def get_batch(self, keys):
        batch = []
        for key in keys:
            data = self.shelve[key]
            batch.append(data)
        batch = np.asarray(batch)
        return batch

    def close(self):
        self.shelve.close()


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
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    parser.add_argument(
        '--test_long',
        help=
        'Test the code by creating an hdf5 file. Take 2 arguments: the number of elements to store and the size of the element',
        type=int,
        nargs=2)
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
        filename = 'test.shelve'
        if os.path.exists(filename):
            os.remove(filename)
        n = args.test_long[0]
        s = args.test_long[1]
        sset = Shelveset(filename)
        timer.start(message=f'# writing {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = random_key()
            v = np.random.uniform(size=s)
            sset.add(k, v)
        timer.stop()
        sset.close()
        sset = Shelveset(filename)
        timer.start(message='# reading keys')
        keys = list(sset.get_keys())
        timer.stop()
        random.shuffle(keys)
        timer.start(message=f'# reading {n} data with size {s} ...')
        for i in tqdm(range(n)):
            k = keys[i]
            v = sset.get(k)
        timer.stop()
        sset.close()
        sys.exit()
