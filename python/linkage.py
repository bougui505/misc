#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import gzip
import os

import numpy as np
import scipy.cluster.hierarchy as hierarchy


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


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
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "npy",
        help="Input npy file containing a condensed distance matrix. For each i and j (where i<j<m), where m is the number of original observations. The metric d_(ij) is stored in entry m * i + j - ((i + 2) * (i + 1)) // 2",
    )
    parser.add_argument(
        "--nclusters", help="Number of clusters to define", type=int)
    parser.add_argument(
        "--distance", help="Distance based clusters", type=float)
    parser.add_argument(
        "--outrec", help="out rec filename (rec.gz) to store the cluster information"
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument(
        "--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    basename = os.path.basename(args.npy)
    if not os.path.exists("linkage"):
        os.mkdir("linkage")
    LINKAGENPY = "linkage/" + os.path.splitext(basename)[0] + "_linkage.npy"
    if os.path.exists(LINKAGENPY):
        print(f"Loading linkage matrix from {LINKAGENPY}")
        Z = np.load(LINKAGENPY)
    else:
        y = np.load(args.npy)
        print(f"{y=}")
        Z = hierarchy.linkage(y, method="average")
        print(f"{Z=}")
        np.save(LINKAGENPY, Z)
    clusters = None
    if args.nclusters is not None:
        CLUSTERREC = (
            "linkage/"
            + os.path.splitext(basename)[0]
            + f"_{args.nclusters}_clusters.rec.gz"
        )
        assert not os.path.exists(
            CLUSTERREC), f"{CLUSTERREC} file already exists"
        clusters = hierarchy.fcluster(
            Z=Z, t=args.nclusters, criterion="maxclust")
    if args.distance is not None:
        CLUSTERREC = (
            "linkage/"
            + os.path.splitext(basename)[0]
            + f"_{args.distance}_clusters.rec.gz"
        )
        assert not os.path.exists(
            CLUSTERREC), f"{CLUSTERREC} file already exists"
        clusters = hierarchy.fcluster(
            Z=Z, t=args.distance, criterion="distance")
    if clusters is not None:
        nclusters = len(np.unique(clusters))
        print(f"{nclusters=}")
        if args.outrec is not None:
            CLUSTERREC = args.outrec
        with gzip.open(CLUSTERREC, "wt") as gz:
            for i, cluster in enumerate(clusters):
                gz.write(f"{i=}\n")
                gz.write(f"{cluster=}\n")
                gz.write("--\n")
