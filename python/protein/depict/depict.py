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
import os
import numpy as np
from pymol import cmd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from descartes import PolygonPatch
from misc.protein import coords_loader
from matplotlib.pyplot import cm
import colorsys
import tqdm


def binarize_z(coords, nbins):
    """
    >>> coords = np.asarray([(i, i, i) for i in range(100)], dtype=float)
    >>> inds = binarize_z(coords, 20)
    >>> inds
    array([ 1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,
            4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  7,  7,
            7,  7,  7,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9, 10, 10, 10, 10,
           10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
           14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17,
           17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20])
    """
    zmin = coords.min(axis=0)[2]
    zmax = coords.max(axis=0)[2]
    bins = np.linspace(zmin, zmax, nbins)
    inds = np.digitize(coords[:, 2], bins)
    return inds


def arr_tuple(li):
    """
    >>> li = [(1,2),(3,4)]
    >>> out = arr_tuple(li)
    >>> out
    array([(1, 2), (3, 4)], dtype=object)
    """
    out = np.empty(len(li), dtype=object)
    out[:] = li
    return out


def get_mapping(keys):
    mapping = np.unique(keys)
    nchain = len(mapping)
    colors = iter(cm.tab20(np.linspace(0, 1, nchain)))
    mapping = {c: next(colors) for c in mapping}
    return mapping


def get_polygons(coords, ax, edgecolor=[0, 0, 0, 1], facecolor=None, zorder=None, alpha=1., label=None):
    polygons = [Point(c[0], c[1]).buffer(1.5) for c in coords]
    u = unary_union(polygons)
    patch2b = PolygonPatch(u, alpha=alpha, ec=edgecolor, fc=facecolor, zorder=zorder, label=label)
    ax.add_patch(patch2b)


def desaturate(rgbcolor, saturation_ratio=1.):
    r, g, b, alpha = rgbcolor
    hsvcolor = list(colorsys.rgb_to_hsv(r, g, b))
    hsvcolor[1] = hsvcolor[1] * saturation_ratio
    h, s, v = hsvcolor
    rgbcolor = list(colorsys.hsv_to_rgb(h, s, v))
    rgbcolor.append(alpha)
    return rgbcolor


def plot_spheres(coords, n_zlevels=20, keys=None, dolegend=False):
    """
    >>> coords = coords_loader.get_coords('1ycr')
    Fetching 1ycr from the PDB
    >>> plot_spheres(coords)
    """
    if keys is None:
        keys = np.ones(len(coords), dtype=int)
    keys = np.asarray(keys)

    inds = binarize_z(coords, nbins=n_zlevels)
    mapping = get_mapping(keys)
    inds = arr_tuple(list(zip(inds, keys)))
    fig = plt.figure()
    ax = fig.add_subplot()
    coords_ = []
    # print(inds)  # [(2, 'A') (2, 'A') (2, 'A') ... (14, 'C') (14, 'C') (14, 'C')]
    unique_inds = np.unique(inds)
    pbar = tqdm.tqdm(total=len(unique_inds))
    pbar.set_description(desc='rendering')
    labels = []
    for ind in unique_inds[::-1]:
        zlevel, key = ind
        sel = np.asarray([e == ind for e in inds])
        coords_ = coords[sel]
        saturation_ratio = zlevel / n_zlevels
        color = mapping[key]
        color = desaturate(color, saturation_ratio)
        edgecolor = [0, 0, 0, 1]
        edgecolor = desaturate(edgecolor, saturation_ratio)
        if key not in labels:
            labels.append(key)
            label = key
        else:
            label = None
        get_polygons(coords_, ax, facecolor=color, edgecolor=edgecolor, zorder=np.median(coords_[:, 2]), label=label)
        pbar.update(1)
    pbar.close()
    ax.set_xlim(coords.min(axis=0)[0] - 2., coords.max(axis=0)[0] + 2.)
    ax.set_ylim(coords.min(axis=0)[1] - 2., coords.max(axis=0)[1] + 2.)
    ax.set_aspect("equal")
    if dolegend:
        # Sort the legend
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)
    if not args.axis:
        plt.axis('off')
    plt.show()


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
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-s', '--sel', default='all')
    parser.add_argument('--zlevels', help='Number of zlevels for bining z-axis (default=20)', default=20, type=int)
    parser.add_argument(
        '--view',
        help=
        "Define the orientation of the protein. In Pymol, first reset the view ('reset' command) then in editing mode apply transformation (shift mouse to apply a transformation) then get the transformation matrix using: 'm = cmd.get_object_matrix('obj_name'); print(m)'"
    )
    parser.add_argument('--axis', help='Display axis', action='store_true')
    parser.add_argument('--legend', help='Display the legend', action='store_true')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    coords, sel = coords_loader.get_coords(args.pdb, selection=args.sel, return_selection=True, view=args.view)
    chains = coords_loader.get_chain_ids(sel)
    plot_spheres(coords, n_zlevels=args.zlevels, keys=chains, dolegend=args.legend)
