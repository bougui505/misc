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


def newrec(line):
    """
    Check if the line is empty to define a new record
    """
    return len(line.strip()) == 0


def get_item(line):
    """
    Return the (key, val) tuple
    """
    key, val = line.split(": ")
    val = val.strip()
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
        except ValueError:
            pass
    return key, val


def add_item(key, val, indict):
    """
    add the item. Handle list
    """
    if key in indict:
        if isinstance(indict[key], list):
            indict[key].append(val)
        else:
            indict[key] = [val, ]
    else:
        indict[key] = val
    return indict


def load(recfilename):
    with open(recfilename, 'r') as recfile:
        pyrec = [dict(), ]
        for i, line in enumerate(recfile):
            if i == 0:
                assert newrec(line) is False  # No empty line at the beginning of the file
            if newrec(line):
                pyrec.append(dict())
            else:
                key, val = get_item(line)
                pyrec[-1] = add_item(key, val, pyrec[-1])
        if len(pyrec[-1]) == 0:
            pyrec.pop()
    return pyrec


if __name__ == '__main__':
    from IPython.terminal.embed import InteractiveShellEmbed
    import argparse
    import pandas as pd
    import os
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-i', '--inp', help='Input recfile', required=True)
    parser.add_argument('-s', '--script', help='Script file to run or to create', required=True)
    args = parser.parse_args()

    global rec
    rec = load(args.inp)
    rec = pd.DataFrame(rec)
    if not os.path.exists(args.script):
        ipshell = InteractiveShellEmbed()
        print('rec file data stored in recfile')
        ipshell.magic(f"%logstart {args.script}")
        ipshell()
    else:
        with open(args.script, 'r') as script:
            cmd = script.read()
        exec(cmd)
        if rec.index.name is None:
            rec.index.name = 'recid'
        print(rec.to_csv())
