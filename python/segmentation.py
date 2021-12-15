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
from pymol import cmd
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os


def get_grid_coords(corfile):
    """
    Input: a cavity cor file
    Returns: a npy array with coordinates of grid points
    """
    cmd.load(filename=corfile, object='cavity')
    grid_coords = cmd.get_coords(selection='cavity')
    return grid_coords


def get_grid(grid_coords):
    """
    Input: grid_coords, a npy array with coordinates of grid points
    Returns: a grid
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = zip(coords.min(axis=0),
                                                   coords.max(axis=0))
    xbins = np.arange(start=xmin, stop=xmax + .5, step=0.5)
    ybins = np.arange(start=ymin, stop=ymax + .5, step=0.5)
    zbins = np.arange(start=zmin, stop=zmax + .5, step=0.5)
    i_list = np.digitize(grid_coords[:, 0], xbins)
    j_list = np.digitize(grid_coords[:, 1], ybins)
    k_list = np.digitize(grid_coords[:, 2], zbins)
    n = i_list.max() + 1
    p = j_list.max() + 1
    q = k_list.max() + 1
    grid = np.zeros((n, p, q))
    grid[(i_list, j_list, k_list)] = 1
    return grid, xbins, ybins, zbins


def get_watershed(grid):
    edt = ndimage.distance_transform_edt(grid)
    sources = peak_local_max(image=edt,
                             footprint=ndimage.generate_binary_structure(3, 2),
                             min_distance=10)
    mask = np.zeros(edt.shape, dtype=bool)
    mask[tuple(sources.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-edt, markers, mask=grid == 1)
    print(np.unique(labels))
    return labels


def write_cor_file(coords, outfilename):
    with open(outfilename, 'w') as outfile:
        npts = len(coords)
        outfile.write(f'{npts}\n')
        for x, y, z in coords:
            outfile.write(
                f'    1    1 DUM  DUM  {x:>9.5f} {y:>9.5f} {z:>9.5f} VAC     1   1.00000\n'
            )


def save_pockets(coords, labels, xbins, ybins, zbins):
    try:
        os.mkdir('cavities')
    except FileExistsError:
        print('cavities directory already exists. Will overwrite')
        pass
    for label in np.unique(labels):
        ilist, jlist, klist = np.where(labels == label)
        x, y, z = xbins[ilist - 1], ybins[jlist - 1], zbins[klist - 1]
        coords = np.c_[x, y, z]
        write_cor_file(coords, f'cavities/cavity_{label}.cor')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-c', '--cor')
    args = parser.parse_args()

    coords = get_grid_coords(args.cor)
    print(coords.shape)
    grid, xb, yb, zb = get_grid(coords)
    print(grid.sum())
    labels = get_watershed(grid)
    save_pockets(coords, labels, xb, yb, zb)
