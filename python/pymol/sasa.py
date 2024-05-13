#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Mar 18 17:47:13 2024

import os

from misc.protein import coords_loader
from pymol import cmd


def load_pdb(pdb, selection, fetch_path=os.path.expanduser("~/pdb")):
    """
    >>> pdb = '1t4e'
    >>> obj = load_pdb(pdb)
    >>> obj # random string
    '...'
    >>> cmd.delete(obj)
    """
    coords, sel = coords_loader.get_coords(
        pdb, selection=selection, return_selection=True, verbose=False, fetch_path=fetch_path
    )
    return sel

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
    parser = argparse.ArgumentParser(description="Compute the sasa relative (see: https://pymolwiki.org/index.php/Get_sasa_relative)")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-p", "--pdb")
    parser.add_argument("-s", "--select")
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

    if args.pdb is not None:
        if args.select is None:
            selection = "all"
        else:
            selection = args.select
        sel = load_pdb(args.pdb, selection=selection)
        sasa = cmd.get_sasa_relative(sel)
        for k in sasa:
            print(k[-1], k[-2], sasa[k])
