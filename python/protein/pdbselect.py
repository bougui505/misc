#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#############################################################################

import os
import subprocess
import tempfile

from pymol import cmd


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
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-s', '--select')
    parser.add_argument('-o', '--out')
    parser.add_argument('--h_add', help="Add hydrogens", action='store_true')
    parser.add_argument('--h_remove', help="Remove hydrogens", action='store_true')
    parser.add_argument('--dockprep', help="Run chimera dockprep command to sanitize, compute charges, ..., before writing a mol2 file", action='store_true')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()

    cmd.load(args.pdb, object='INPDB')
    if args.h_remove:
        cmd.remove("hydrogens")
    if args.h_add:
        cmd.h_add("all")
    if args.dockprep:
        # Save a temporary pdb as chimera will interpret the atom type. It does not when a mol2 is given
        pdb_tmp = tempfile.NamedTemporaryFile(suffix='.pdb').name
        cmd.save(pdb_tmp, selection=args.select)
        subprocess.run(f"{GetScriptDir()}/dockprep.sh -i {pdb_tmp} -o {args.out}", shell=True)
        print("dockprep done")
    else:
        cmd.save(args.out, selection=args.select)
