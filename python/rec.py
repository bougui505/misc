#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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
import collections


def checklengths(data, fields):
    # check if the number of lines is consistent:
    lengths = []
    for field in fields:
        lengths.append(len(data[field]))
    maxl = max(lengths)
    # Extend with "-" if needed
    for field in fields:
        if len(data[field]) < maxl:
            data[field].extend(["-"] * (maxl - len(data[field])))
    return data


def print_data(recsel=None, print_records=False):
    if recsel is not None:
        print(f"{recsel=}")
        print("--")
    data = collections.defaultdict(list)
    current_record = ""
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line != "--":
            current_record += line + "\n"
            kv = line.split("=", maxsplit=1)
            if len(kv) != 2:
                continue
            key, value = kv
            if key in args.fields:
                if recsel is not None:
                    exec(f"{key}={value}")
                    store = eval(recsel)
                else:
                    store = True
                if store:
                    data[key].append(value)
        else:
            data = checklengths(data, args.fields)
            if print_records and store:
                print(current_record + "--")
            current_record = ""
    data = checklengths(data, args.fields)
    n = max(len(v) for _, v in data.items())
    header = [f"#{e}" for e in args.fields]
    header = " ".join(header)
    if not print_records:
        print(header)
        for i in range(n):
            outstr = ""
            for key in args.fields:
                outstr += data[key][i] + " "
            print(outstr)


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

    parser = argparse.ArgumentParser(
        description="Read a python like recfile from stdin (pipe)"
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--info", help="Print long help message.", action="store_true")
    parser.add_argument(
        "-f", "--fields", help="Fields to extract", nargs="+", default=[]
    )
    parser.add_argument(
        "-s",
        "--sel",
        help="Selection string for the extracted field (see: --fields). E.g. 'a>2.0', where 'a' is a field key",
    )
    parser.add_argument(
        "-r",
        "--print_records",
        action="store_true",
        help="Print the selected records instead of the data",
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.info:
        sys.stdout.write(
            """\
Read a python like recfile. A pyrec file looks like:

# comment1
field1=1
field2=2
# comment2
field3=3
--
# comment3
field2=5
field1=4
field3=6

The easiest way to create such a file format is to use:

print(f"{var=:.4g}")

        """
        )
        sys.exit()

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

    print_data(recsel=args.sel, print_records=args.print_records)
