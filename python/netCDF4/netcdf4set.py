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
from netCDF4 import Dataset
import numpy as np


class NetCDF4set(object):
    """
    >>> n4filename = '/tmp/test.nc'
    >>> if os.path.exists(n4filename):
    ...     os.remove(n4filename)
    >>> shape = (3, 3)
    >>> n4set = NetCDF4set(n4filename, shape=shape)
    >>> n4set.add('a', np.ones(shape))
    >>> n4set.add_batch(['b', 'c', 'd'], np.stack((np.ones(shape)*2, np.ones(shape)*3, np.ones(shape)*4)))
    >>> n4set.add('a', np.random.uniform(size=(shape)))
    # key "a" already exists in /tmp/test.nc
    >>> n4set.get_keys()
    dict_keys(['a', 'b', 'c', 'd'])
    >>> n4set.get('a')
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    >>> n4set.get('b')
    array([[2., 2., 2.],
           [2., 2., 2.],
           [2., 2., 2.]], dtype=float32)
    >>> n4set.get('c')
    array([[3., 3., 3.],
           [3., 3., 3.],
           [3., 3., 3.]], dtype=float32)
    """
    def __init__(self, n4filename, shape, mode='w'):
        """
        """
        self.n4filename = n4filename
        self.n4file = Dataset(n4filename, mode, format="NETCDF4")
        self.dimensions = self.create_dimensions(shape)

    def get_keys(self):
        keys = self.n4file.variables.keys()
        return keys

    def create_dimensions(self, shape):
        dimensions = []
        for i, dim in enumerate(shape):
            i = str(i)
            self.n4file.createDimension(i, dim)
            dimensions.append(i)
        return tuple(dimensions)

    def add_batch(self, keys, batch, datatype='f4'):
        for i, key in enumerate(keys):
            data = batch[i]
            self.add(key, data, datatype=datatype)

    def add(self, key, data, datatype='f4'):
        if key not in self.n4file.variables:
            n4var = self.n4file.createVariable(key, datatype, dimensions=self.dimensions)
            n4var[:] = data
        else:
            print(f'# key "{key}" already exists in {self.n4filename}')

    def get(self, key):
        n4var = self.n4file.variables[key]
        return n4var[:].data


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
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
