#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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
from misc import moving_average
from scipy import signal
import scipy.spatial.distance as scidist
import matplotlib.pyplot as plt


class Spectrogram(object):
    def __init__(self, data, moving_average_window=None):
        self.data = data
        self.moving_average_window = moving_average_window
        self._get()

    def _get(self):
        if self.moving_average_window is not None:
            self.baseline = moving_average.apply(self.data,
                                                 self.moving_average_window)
            self.data -= self.baseline
        self.frequencies, self.time, self.Sxx = signal.spectrogram(self.data,
                                                                   fs=1.)
        if self.Sxx.ndim == 2:
            self.Sxx = self.Sxx[None, ...]
        self.ncells, self.nfreq, self.nt = self.Sxx.shape

    def plot(self, labels=None, correlation=False):
        X = self.time
        if correlation:
            data = self.correlation()
            Y = np.arange(1, data.shape[1] + 2)
        else:
            data = self.Sxx
            Y = self.frequencies
        nplots = data.shape[0]
        nrows = int(np.sqrt(nplots))
        ncols = nrows + nplots % nrows
        f, axes = plt.subplots(nrows, ncols, sharex='all', sharey='all')
        for i, data_ in enumerate(data):
            ip, jp = np.unravel_index(i, (nrows, ncols))
            ax = axes[ip, jp]
            pcm = ax.pcolormesh(X,
                                Y,
                                data_,
                                shading='flat',
                                cmap='binary',
                                vmin=0.)
            f.colorbar(pcm, ax=ax)
            if ip == nrows - 1:
                ax.set_xlabel('Time')
            if jp == 0:
                ax.set_ylabel('Frequency')
            if labels is not None:
                ax.set_title(labels[i], fontsize=14)
        todel = range(i + 1, nplots + 2)
        for j in todel:
            ip, jp = np.unravel_index(j, (nrows, ncols))
            f.delaxes(axes[ip][jp])
        plt.show()

    def correlation(self):
        """
        Time resolve correlation
        """
        print(self.Sxx.shape)
        corr = np.zeros((self.ncells, self.ncells, self.nt))
        for t in range(self.nt):
            for cell1 in range(self.ncells):
                spectrum1 = self.Sxx[cell1, :, t]
                for cell2 in range(self.ncells):
                    spectrum2 = self.Sxx[cell2, :, t]
                    corrval = 1. - scidist.correlation(spectrum1, spectrum2)
                    corr[cell1, cell2, t] = corrval
        return corr


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()
