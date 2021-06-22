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

from sklearn.decomposition import PCA
import numpy as np
from scipy.interpolate import griddata
# from scipy.stats import binned_statistic_2d


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(np.c_[x, y], z, (X, Y), method='linear')
    # Z, xi, yi, _ = binned_statistic_2d(x, y, z, bins=(100, 100), statistic='max')
    # X, Y = np.meshgrid(xi[:-1], yi[:-1])
    return X, Y, Z


class Miller(object):
    """
    see: https://bmcstructbiol.biomedcentral.com/articles/10.1186/s12900-016-0055-7#Sec1
    """
    def __init__(self):
        self.pca = PCA(n_components=3)

    def fit(self, X):
        self.pca.fit(X)
        X = self.pca.transform(X)
        self.r = np.linalg.norm(X, axis=1).max()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        X = self.pca.transform(X)
        self.alt = np.linalg.norm(X, axis=1)
        X = self.r * X / self.alt[:, None]
        lat = np.arctan(X[:, 2] / self.r)
        lon = np.arctan(X[:, 1] / X[:, 0])
        xp = lon
        yp = (5 / 4) * np.log(np.tan(np.pi / 4 + (2 / 5) * lat))
        xy = np.c_[xp, yp]
        return xy


if __name__ == '__main__':
    import pdbsurf
    import matplotlib.pyplot as plt
    import recutils
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-s', '--sel')
    parser.add_argument('-c', '--caption')
    parser.add_argument('--size', type=float, help='Size of the dots for the scatter plot (default=1.)', default=1.)
    parser.add_argument('--levels', type=int, help='Number of levels in the contour plot (default=10)', default=10)
    args = parser.parse_args()

    miller = Miller()
    surfpts = pdbsurf.pdb_to_surf(args.pdb, args.sel)
    proj = miller.fit_transform(surfpts)
    # plt.scatter(proj[:, 0], proj[:, 1], s=8, c=miller.alt, cmap='gist_gray')
    X, Y, Z = grid(proj[:, 0], proj[:, 1], miller.alt)
    plt.contourf(X, Y, Z, cmap='coolwarm', levels=args.levels)
    plt.colorbar()
    if args.caption is not None:
        captions = recutils.load(args.caption)
        for caption in captions:
            sel = caption['sel']
            color = caption['color']
            print(f'{color}: {sel}')
            surfpts = pdbsurf.pdb_to_surf(args.pdb, sel)
            proj_ = miller.transform(surfpts)
            plt.scatter(proj_[:, 0], proj_[:, 1], s=args.size, color=color)
    plt.show()
