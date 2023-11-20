#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
import os

import numpy as np
import scipy.spatial.distance as scidist
import torch


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def py2txt(pyfile, delimiter=" ", squareform=False):
    ext = os.path.splitext(pyfile)[-1]
    if ext == ".pt":
        data = torch.load(pyfile, map_location=torch.device("cpu"))
        data = data.numpy()
    elif ext == ".npy":
        data = np.load(pyfile)
    else:
        sys.exit("Only .pt or .npy extensions are known")
    if squareform:
        data = scidist.squareform(data)
    n = data.shape[0]
    for i in range(n):
        line = data[i]
        if hasattr(line, "__iter__"):
            string = delimiter.join([str(e) for e in line])
        else:
            string = line
        print(string)


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
        description="Read a PyTorch or numpy file (.pt or .npy) and print it as text on the stdout."
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "inp", help="Input PyTorch file (.pt) or numpy file (.npy) to convert in txt"
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        help="Column delimiter to use (default 1-space)",
        default=" ",
    )
    parser.add_argument(
        "--squareform",
        help="Apply scipy.spatial.distance.squareform before printing. See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html",
        action="store_true",
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
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

    if args.inp is not None:
        py2txt(args.inp, args.delimiter, squareform=args.squareform)
