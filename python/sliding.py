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

import numpy as np
import matplotlib.pyplot as plt


class Sliding_op(object):
    """
    See: https://stackoverflow.com/a/40773366/1679629

    a: input data (1D)
    window_size: size of the sliding window
    func: function to apply on the window. The function must take an array as input and an axis argument

    Without padding
    >>> a = np.random.normal(size=100)
    >>> a += np.sin(0.25*np.arange(100)) * 3.
    >>> slmean = Sliding_op(a, 10, np.mean)
    >>> slmean.transform().shape
    (91,)
    >>> slmean.plot()

    With padding
    >>> a = np.random.normal(size=100)
    >>> a += np.sin(0.25 * np.arange(100)) * 3.
    >>> slmean = Sliding_op(a, 10, np.mean, padding=True)
    >>> slmean.transform().shape
    (100,)
    >>> slmean.plot()
    """
    def __init__(self, a, window_size, func, padding=False):
        self.a = a
        self.window_size = window_size
        self.func = func
        self.npadleft, self.npadright = 0, 0
        if padding:
            self.pad()
        self.outlen = self.a.size - self.window_size + 1

    def pad(self):
        """
        pad the data with random number from a normal distribution
        """
        nout = self.a.size - self.window_size + 1
        npad = self.a.size - nout
        self.npadleft = 0
        self.npadright = npad - self.npadleft
        mu, sigma = self.a[:self.window_size].mean(), self.a[:self.window_size].std()
        padleft = np.random.normal(loc=mu, scale=sigma, size=self.npadleft)
        mu, sigma = self.a[-self.window_size:].mean(), self.a[-self.window_size:].std()
        padright = np.random.normal(loc=mu, scale=sigma, size=self.npadright)
        self.a = np.r_[padleft, self.a, padright]

    def transform(self):
        n = self.a.strides[0]
        a2D = np.lib.stride_tricks.as_strided(self.a, shape=(self.outlen, self.window_size), strides=(n, n))
        out = self.func(a2D, axis=1)
        return out

    def unpad(self):
        if self.npadright > 0:
            return self.a[self.npadleft:-self.npadright]
        else:
            return self.a[self.npadleft:]

    def plot(self):
        plt.plot(self.unpad())
        out = self.transform()
        plt.plot(out, color='red')
        plt.show()


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
        sys.exit()
