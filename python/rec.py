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
import numpy as np
import scipy.spatial.distance as scidist
import re


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


def read_file(
    file=None,
    fields=None,
    recsel=None,
    print_records=False,
    print_columns=True,
    delimiter=" ",
):
    """
    delimiter: delimiter between columns for printing output
    """
    if isinstance(file, str):
        file = open(file, "r")
    if recsel is not None and print_records:
        print(f"{recsel=}")
        print("--")
    if fields is None or len(fields) == 0:
        fields = []
        return_all_fields = True
    else:
        return_all_fields = False
    # Use a dictionnary eith key to None to have an ordered set for fields
    # see: https://stackoverflow.com/a/53657523/1679629
    fields = dict(zip(fields, [None] * len(fields)))
    data = collections.defaultdict(list)
    for line in file:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line != "--":
            kv = line.split("=", maxsplit=1)
            if kv == [""]:
                continue
            assertstr = f"key, value couple needed -- {kv=}"
            assert len(kv) == 2, assertstr
            key, value = kv
            if key in fields or return_all_fields:
                if return_all_fields and key not in fields:
                    fields[key] = None
                data[key].append(value)
        else:
            data = checklengths(data, fields)
    data = checklengths(data, fields)
    data = listdict_to_arrdict(data)
    if recsel is not None:
        data = data_selection(data, recsel)
    n = max(len(v) for _, v in data.items())
    header = [f"#{e}" for e in fields]
    header = " ".join(header)
    if not print_records and print_columns:
        print(header)
        for i in range(n):
            outstr = ""
            for key in fields:
                outstr += str(data[key][i]) + delimiter
            print(outstr)
    else:
        dict_to_rec(data)
    return data


def is_float(element) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def convert_to_array(l: list) -> np.ndarray:
    """
    >>> l = [1, 2, 3, '4', '-', 5, 6]
    >>> convert_to_array(l)
    array([ 1.,  2.,  3.,  4., nan,  5.,  6.])
    >>> l = ['a', 'b', 'c']
    >>> convert_to_array(l)
    array(['a', 'b', 'c'], dtype='<U1')
    """
    l1 = [float(e) if is_float(e) else np.nan for e in l]
    arr = np.asarray(l1, dtype=float)
    if np.isnan(arr).all():
        return np.asarray(l)
    else:
        return arr


def listdict_to_arrdict(d: dict) -> dict:
    for k in d.keys():
        d[k] = convert_to_array(d[k])
    return d


def data_selection(data: dict, recsel: str) -> dict:
    """"""
    keys = list(data.keys())
    lengths = np.asarray([len(data[k]) for k in keys])
    assert (lengths[0] == lengths).all()
    n = lengths[0]
    out = collections.defaultdict(list)
    for i in range(n):
        for key in data:
            vars()[key] = data[key][i]
            # exec(f"{key}={data[key][i]}")
        keep = eval(recsel)
        if keep:
            for key in data:
                out[key].append(data[key][i])
    return out


def merge_dictionnaries(d1, d2):
    """"""
    values1 = []
    values2 = []
    keys1 = list(d1.keys())
    keys2 = list(d2.keys())
    common_keys = set(keys1) & set(keys2)
    for k in common_keys:
        values1.append(d1[k])
        values2.append(d2[k])
    values1 = np.asarray(values1).T
    values2 = np.asarray(values2).T
    # Compute pairwise equality peq
    peq = scidist.cdist(values1, values2, metric=lambda x, y: (x == y).all())
    peq = peq.astype(bool)
    inds = np.where(peq)
    out = collections.defaultdict(list)
    for i1, i2 in zip(*inds):
        for k in common_keys:
            v1 = d1[k][i1]
            v2 = d2[k][i2]
            assert v1 == v2
            out[k].append(v1)
        for k in set(keys1) - common_keys:
            out[k].append(d1[k][i1])
        for k in set(keys2) - common_keys:
            out[k].append(d2[k][i2])
    return out


def dict_to_rec(d):
    """
    Print a dictionnary as a rec file
    """
    keys = list(d.keys())
    nval = len(d[keys[0]])
    for i in range(nval):
        for k in keys:
            v = d[k][i]
            print(f"{k}={v}")
        print("--")


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

    parser = argparse.ArgumentParser(
        description="Read a python like recfile from stdin (pipe) except if --file is given"
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--info", help="Print long help message.", action="store_true")
    parser.add_argument(
        "-f",
        "--fields",
        help="Fields to extract. If no fields are given extract all fields.",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        help="Delimiter between columns to print output (default: ' ')",
        default=" ",
    )
    parser.add_argument(
        "-s",
        "--sel",
        help="Selection string for the extracted field (see: --fields). E.g. 'a>2.0', where 'a' is a field key",
    )
    parser.add_argument(
        "--find", help="Find a substring in a string. The syntax is 'substr in field'"
    )
    parser.add_argument(
        "-r",
        "--print_records",
        action="store_true",
        help="Print the selected records instead of the data",
    )
    parser.add_argument(
        "--file",
        help="By default, read from stdin. If a file is given read from the given file",
    )
    parser.add_argument(
        "--merge",
        help="Merge the two given files, based on the common fields",
        nargs=2,
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

    if args.find is not None:
        # remove extra spaces:
        args.find = re.sub(" +", " ", args.find)
        argssplit = args.find.split(" not in ")
        NEGATION = True
        if len(argssplit) == 1:
            argssplit = args.find.split(" in ")
            NEGATION = False
        assertstr = f"Cannot interpret find expression: {args.find}"
        assert len(argssplit) == 2, assertstr
        substr = argssplit[0].strip()
        field = argssplit[1].strip()
        if not NEGATION:
            findexpr = f"{field}.find('{substr}')!=-1"
        else:
            findexpr = f"{field}.find('{substr}')==-1"
        if args.sel is None:
            args.sel = findexpr
        else:
            args.sel = f"{args.sel} and {findexpr}"

    if args.merge is not None:
        DATA1 = read_file(
            file=args.merge[0],
            fields=None,
            recsel=args.sel,
            print_records=False,
            print_columns=False,
        )
        DATA2 = read_file(
            file=args.merge[1],
            fields=None,
            recsel=args.sel,
            print_records=False,
            print_columns=False,
        )
        MERGED = merge_dictionnaries(DATA1, DATA2)
        dict_to_rec(MERGED)
        sys.exit(0)
    if args.file is None:
        args.file = sys.stdin
    DATA = read_file(
        file=args.file,
        fields=args.fields,
        recsel=args.sel,
        print_records=args.print_records,
        delimiter=args.delimiter,
    )
