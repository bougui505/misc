#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Dec  5 10:55:05 2023
#
# Awk like text processing using python

import os
import select
import sys

import numpy as np


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir

def print_(indata):
    try:
        indata[indata == None] = '_'
    except (TypeError, ValueError):
        pass
    if args.delimiter is None:
        d = ' '
    else:
        d = args.delimiter
    if indata.ndim == 1:
        sys.stdout.write(f'{d.join([str(e) for e in indata])}\n')
    elif indata.ndim == 2:
        for line in indata:
            sys.stdout.write(f'{d.join([str(e) for e in line])}\n')
    elif indata.ndim == 0:
        sys.stdout.write(str(indata) + '\n')

def format_line(line):
    line = line.split(args.delimiter)
    outline = []
    for e in line:
        e = e.strip()
        try:
            e = int(float(e)) if int(float(e)) == float(e) else float(e)
        except ValueError:
            pass
        outline.append(e)
    outline = np.asarray(outline)
    return outline


if __name__ == "__main__":
    import argparse
    import doctest

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
    parser = argparse.ArgumentParser(prog="pyawk",
                                     description="awk like text processing using python",
                                     usage="paste -d ' ' =(seq 10) =(seq 11 20) | ./pyawk.py 'print(d[1],d[0])'")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-d", "--delimiter", default=" ")
    parser.add_argument("--info", help="Detailed help message", action='store_true')
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    parser.add_argument('cmd', help='Command to run')
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

    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        # sys.stdin is not empty
        for line in sys.stdin:
            line = line.strip()
            d = format_line(line)
            exec(args.cmd)
