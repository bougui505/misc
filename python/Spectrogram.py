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

    def plot(self, labels=None):
        if self.Sxx.ndim == 2:
            self.Sxx = self.Sxx[None, ...]
        nplots = self.Sxx.shape[0]
        nrows = int(np.sqrt(nplots))
        ncols = nrows + nplots % nrows
        print(nplots, nrows, ncols)
        f, axes = plt.subplots(nrows, ncols, sharex='all', sharey='all')
        for i, Sxx in enumerate(self.Sxx):
            ip, jp = np.unravel_index(i, (nrows, ncols))
            ax = axes[ip, jp]
            ax.pcolormesh(self.time,
                          self.frequencies,
                          Sxx,
                          shading='flat',
                          cmap='binary')
            if ip == nrows - 1:
                ax.set_xlabel('Time')
            if jp == 0:
                ax.set_ylabel('Frequency')
            if labels is not None:
                ax.set_title(labels[i])
        todel = range(i + 1, nplots + 2)
        for j in todel:
            ip, jp = np.unravel_index(j, (nrows, ncols))
            f.delaxes(axes[ip][jp])
        plt.show()


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()
