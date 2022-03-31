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
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from misc import sliding


class Local_peaks(object):
    """
    # Detect local maxima
    >>> data = np.random.normal(size=100)
    >>> data += np.sin(0.25*np.arange(100)) * 3.
    >>> data[[10, 40, 80, 90]] += 20.
    >>> local_peaks = Local_peaks(data, zscore=None, wlen=10, logging=logging)
    >>> local_peaks.peaks
    array([10, 40, 80, 90])
    >>> local_peaks.plot()

    # Detect local minima
    >>> data = np.random.normal(size=100)
    >>> data += np.sin(0.25*np.arange(100)) * 3.
    >>> data[[10, 40, 80, 90]] -= 20.
    >>> local_peaks = Local_peaks(data, zscore=None, wlen=10, minima=True, logging=logging)
    >>> local_peaks.peaks
    array([10, 40, 80, 90])
    >>> local_peaks.plot()
    """
    def __init__(self, data, zscore=None, wlen=10, minima=False, logging=None):
        self.data = data
        self.zscore = zscore
        self.wlen = wlen
        self.minima = minima
        self.logging = logging
        if self.logging is not None:
            self.logging.info(f'zscore: {self.zscore}')
            self.logging.info(f'wlen: {self.wlen}')
            self.logging.info(f'minima: {self.minima}')

    def _automatic_threshold(self, zscores, zz=3.):
        thr = abs(zscores.mean()) + zz * zscores.std()
        return thr

    @property
    def peaks(self):
        slmu = sliding.Sliding_op(self.data, window_size=self.wlen, func=np.mean, padding=True).transform()
        slsigma = sliding.Sliding_op(self.data, window_size=self.wlen, func=np.std, padding=True).transform()
        slz = (self.data - slmu) / slsigma
        if self.zscore is None:
            self.zscore = self._automatic_threshold(slz)
            if self.logging is not None:
                self.logging.info(f'Automatic_zcore_threshold: {self.zscore:.3f}')
        if not self.minima:
            if self.logging is not None:
                self.logging.info(f'zscores.max: {slz.max():.3f}')
            peaks = np.where(slz > self.zscore)[0]
        else:
            if self.logging is not None:
                self.logging.info(f'zscores.min: {slz.min():.3f}')
            peaks = np.where(slz < -self.zscore)[0]
        return peaks

    def plot(self):
        peaks = self.peaks
        plt.plot(self.data)
        for peak in peaks:
            plt.axvline(peak, linewidth=1., linestyle='--', color='red')
        plt.show()


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
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
