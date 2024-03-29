#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import os

import numpy as np
import pymol2
from misc.Grid3 import mrc
from misc.protein import coords_loader
from scipy.ndimage.morphology import distance_transform_edt


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_all_coords(pdblist):
    """
    >>> pdblist = ["1ycr", "1t4e"]
    >>> coords = get_all_coords(pdblist)
    Fetching 1ycr from the PDB
    Fetching 1t4e from the PDB
    >>> coords.shape
    (2500, 3)
    """
    all_coords = []
    for pdb in pdblist:
        coords = coords_loader.get_coords(pdb)
        all_coords.append(coords)
    all_coords = np.concatenate(all_coords)
    return all_coords


def get_density(coords, spacing=1.0, sigma=1.7, padding=2.0, normalize=False):
    """
    >>> pdblist = ["1ycr", "1t4e"]
    >>> coords = get_all_coords(pdblist)
    Fetching 1ycr from the PDB
    Fetching 1t4e from the PDB
    >>> coords.shape
    (2500, 3)
    >>> density, edges = get_density(coords)
    >>> density.shape
    (70, 63, 63)
    >>> len(edges)
    3
    >>> (edges[0].shape, edges[1].shape, edges[2].shape,)
    ((71,), (64,), (64,))
    """
    xmin, ymin, zmin = coords.min(axis=0) - padding
    xmax, ymax, zmax = coords.max(axis=0) + padding
    drange = [
        np.arange(start=xmin, stop=xmax, step=spacing),
        np.arange(start=ymin, stop=ymax, step=spacing),
        np.arange(start=zmin, stop=zmax, step=spacing),
    ]
    count, edges = np.histogramdd(coords, bins=drange)
    weights = np.unique(count)
    density = np.zeros(count.shape)
    for w in weights:
        if w != 0:
            edt = distance_transform_edt(~(count == w))
            density += w * np.exp(-(edt**2) / (2 * sigma**2))
    if normalize:
        density -= density.min()
        density /= density.max()
    return density, edges


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

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
    parser = argparse.ArgumentParser(
        description="Compute atomic density from pdb, or related structure files")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "-p",
        "--pdb",
        help="PDB or related structure files. Multiple files accepted.",
        nargs="+",
    )
    parser.add_argument("--traj", help="Optionnal trajectory file")
    parser.add_argument("--selection",
                        help="Selection of atoms",
                        default="all")
    parser.add_argument(
        "-s",
        "--spacing",
        help="spacing of the density grid (default: 1.0 Å)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--padding",
        help="padding of the density grid (default: 2.0 Å)",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--sigma",
        help="sigma for the gaussians (default: 1.7 Å)",
        type=float,
        default=1.7,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the density between 0.0 and 1.0",
    )
    parser.add_argument("-o", "--out", help="output MRC filename")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func",
                        help="Test only the given function(s)",
                        nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS
                            | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS
                    | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    if args.traj is not None:
        with pymol2.PyMOL() as p:
            p.cmd.set("defer_builds_mode", 3)
            p.cmd.load(args.pdb[0])
            p.cmd.load_traj(args.traj)
            coords = p.cmd.get_coords(selection=args.selection, state=0)
            print(f"{coords.shape=}")
    else:
        coords = get_all_coords(pdblist=args.pdb)
    density, edges = get_density(
        coords=coords,
        padding=args.padding,
        spacing=args.spacing,
        sigma=args.sigma,
        normalize=args.normalize,
    )
    print("density.shape:", density.shape)
    origin = [e.min() for e in edges]
    print("origin:", origin)
    mrc.save_density(
        density=density,
        outfilename=args.out,
        spacing=args.spacing,
        origin=origin,
    )
