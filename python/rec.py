#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import collections
import gzip
import os
import re
import subprocess

import numpy as np
import scipy.spatial.distance as scidist
from joblib import Parallel, delayed
from tqdm import tqdm


def format_data(data: dict, fields: list) -> dict:
    """"""
    out = collections.defaultdict(list)
    for recid in data:
        subdict = data[recid]
        for key in fields:
            if key not in subdict:
                out[key].append("-")
            else:
                out[key].append(subdict[key])
    return out


def columnsfile_to_data(file, delimiter=" ") -> dict:
    """
    The first line must be the header:
    #field1 #field2 ...
    """
    if isinstance(file, str):
        file = open(file, "r")
    data = collections.defaultdict(list)
    for linenbr, line in enumerate(file):
        # remove repeated spaces
        line = re.sub(" +", " ", line)
        line = line.strip()
        line = line.split(sep=delimiter)
        if linenbr == 0:
            fields = [e[1:].strip() for e in line]
        else:
            for key, value in zip(fields, line):
                data[key].append(value)
    return data


def get_data(file, selected_fields=None, rmquote=False):
    if isinstance(file, str):
        ext = os.path.splitext(file)[-1]
        if ext == ".gz":
            file = gzip.open(file, "rt")
        else:
            file = open(file, "r")
    data = dict()
    recid = 0  # record id
    fields = set()
    for linenbr, line in tqdm(enumerate(file), desc="reading file..."):
        linenbr += 1
        line = line.strip()
        if line.startswith("#"):
            continue
        if line != "--":
            kv = line.split("=", maxsplit=1)
            if kv == [""]:
                continue
            assertstr = f"key, value couple needed in line {linenbr} -- {kv=}"
            assert len(kv) == 2, assertstr
            key, value = kv
            if len(value.strip()) == 0:  # replace empty str by "-"
                value = "-"
            if rmquote:
                value = value.replace("'", "")
            if selected_fields is not None:
                if key in selected_fields:
                    fields.add(key)
                    if recid not in data:
                        data[recid] = dict()
                    data[recid][key] = value
            else:
                fields.add(key)
                if recid not in data:
                    data[recid] = dict()
                data[recid][key] = value
        else:
            recid += 1
    fields = list(fields)
    if selected_fields is None:
        fields.sort()
    else:
        fields = selected_fields
    data = format_data(data, fields)
    data = listdict_to_arrdict(data)
    return data, fields


def sort(data: dict, field: str) -> dict:
    """"""
    sorter = data[field].argsort()
    for key in data.keys():
        data[key] = data[key][sorter]
    return data


def read_file(
    file=None,
    selected_fields=None,
    recsel=None,
    print_records=False,
    print_columns=True,
    delimiter=" ",
    get_stats=False,
    rmquote=False,
):
    """
    delimiter: delimiter between columns for printing output
    """
    if recsel is not None and print_records:
        print(f"{recsel=}")
        print("--")
    data, fields = get_data(file, selected_fields, rmquote=rmquote)
    if get_stats:
        stat_keys = list(data.keys())
        print(f"keys={stat_keys}")
        nrecords = len(data[stat_keys[0]])
        print(f"{nrecords=}")
        sys.exit(0)
    if recsel is not None and len(data) > 0:
        data = data_selection(data, recsel)
    if len(data) == 0:
        return data
    n = max(len(v) for _, v in data.items())
    header = [f"#{e}" for e in fields]
    header = delimiter.join(header)
    if not print_records and print_columns:
        print(header)
        for i in range(n):
            outstr = ""
            for key in fields:
                outstr += str(data[key][i]) + delimiter
            print(outstr)
    if print_records:
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
    array(['1', '2', '3', '4', '-', '5', '6'], dtype='<U21')
    >>> l = ['a', 'b', 'c']
    >>> convert_to_array(l)
    array(['a', 'b', 'c'], dtype='<U1')
    >>> l = [1,2,3,4,5,6]
    >>> convert_to_array(l)
    array([1, 2, 3, 4, 5, 6])
    >>> l = [1,2,3,4.2,5,6]
    >>> convert_to_array(l)
    array([1. , 2. , 3. , 4.2, 5. , 6. ])
    """
    l1 = [float(e) if is_float(e) else np.nan for e in l]
    arr = np.asarray(l1, dtype=float)
    n_nan = np.isnan(arr).sum()
    r_nan = n_nan / len(arr)
    if r_nan > 0:
        return np.asarray(l)
    else:
        if (np.int_(arr) == arr).all():
            return np.int_(arr)
        else:
            return arr


def listdict_to_arrdict(d: dict) -> dict:
    for k in d.keys():
        d[k] = convert_to_array(d[k])
    return d


def check_data_lengths(data):
    keys = list(data.keys())
    lengths = np.asarray([len(data[k]) for k in keys])
    assert (lengths[0] == lengths).all(), lengths
    return lengths[0]


def data_selection(data: dict, recsel: str) -> dict:
    """"""
    n = check_data_lengths(data)
    out = collections.defaultdict(list)
    n_found = 0
    for rec_internal_index in (pbar := tqdm(range(n))):
        for key in data:
            vars()[key] = data[key][rec_internal_index]
            # exec(f"{key}={data[key][i]}")
        keep = eval(recsel)
        if keep:
            n_found += 1
            pbar.set_description(
                f"nbr of match: {n_found}/{rec_internal_index+1}")
            for key in data:
                out[key].append(data[key][rec_internal_index])
        if rec_internal_index == n - 1:
            pbar.set_description(
                f"nbr of match: {n_found}/{rec_internal_index+1}")
    return out


def add_property(data: dict, property: str, name: str) -> dict:
    """
    Compute the property by interpreting 'property' and add it to data under the name
    """
    n = check_data_lengths(data)
    keys = list(data.keys())
    for i in tqdm(range(n)):
        for key in keys:
            vars()[key] = data[key][i]
        result = eval(property)
        data[name].append(result)
    return data


def run(data: dict, cmd: str, fields: list, names: list) -> dict:
    n = check_data_lengths(data)
    n_jobs = os.cpu_count()
    cmd_list = []
    for i in range(n):
        args_i = []
        cmd_i = [cmd]
        for key in fields:
            args_i.append(str(data[key][i]))
        cmd_i.extend(args_i)
        cmd_list.append(cmd_i)
    out = Parallel(n_jobs=n_jobs)(delayed(subprocess.check_output)(inp)
                                  for inp in tqdm(cmd_list))
    out = [e.strip().decode() for e in out]
    if len(names) == 1:
        name = names[0]
        data[name] = out
    else:
        nout = len(names)
        assert len(out[0].split(" ")) == nout
        out = [e.split(" ") for e in out]
        for i in range(nout):
            name = names[i]
            data[name] = [e[i] for e in out]
    return data


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
    for i1, i2 in tqdm(zip(*inds), total=len(inds), desc="merging"):
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


def dict_to_rec(d, outgz=None):
    """
    Print a dictionnary as a rec file
    """
    keys = list(d.keys())
    nval = len(d[keys[0]])
    if outgz is not None:
        gz = gzip.open(outgz, "wt")
    for i in range(nval):
        for k in keys:
            v = d[k][i]
            if outgz is None:
                print(f"{k}={v}")
            else:
                gz.write(f"{k}={v}\n")
        if outgz is None:
            print("--")
        else:
            gz.write("--\n")
    if outgz is not None:
        gz.close()


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    parser = argparse.ArgumentParser(
        description="Read a python like recfile from stdin (pipe) except if --file is given"
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--info",
                        help="Print long help message.",
                        action="store_true")
    parser.add_argument(
        "-f",
        "--fields",
        help="Fields to extract. If no fields are given extract all fields.",
        nargs="+",
        default=None,
    )
    parser.add_argument("--sort",
                        help="Sort the rec file according to the given field")
    parser.add_argument(
        "-d",
        "--delimiter",
        help="Delimiter between columns to print output (default: ' ')",
        default=" ",
    )
    parser.add_argument(
        "-s",
        "--sel",
        help="Selection string for the extracted field (see: --fields). E.g. 'a>2.0', where 'a' is a field key. Can also be a path to the file containing the selection string.",
    )
    parser.add_argument(
        "--rmquote",
        help="Remove the simple quotes ' for the values. This simplify the selection string. fields='abc' becomes fields=abc",
        action="store_true",
    )
    parser.add_argument(
        "--calc",
        help="Property to compute from the given field and the name to store the result in. E.g. 'y=x*10': add the field y and store the result of field x*10",
    )
    parser.add_argument(
        "--run",
        help="run the given command from the given field (--fields). The syntax is field=cmd. This will store the result of command 'cmd' in the field 'field'. The arguments are given by the --fields option. The running programm can return 2 values, space separated. In this case 2 field names can be given to store the output (e.g. field1,field2=cmd)",
    )
    parser.add_argument(
        "--find",
        help="Find a substring in a string. The syntax is 'substr in field', Multiple find expression can be given",
        nargs="+",
    )
    parser.add_argument(
        "-r",
        "--print_records",
        action="store_true",
        help="Print the selected records instead of the data",
    )
    parser.add_argument(
        "--file",
        help="By default, read from stdin. If a file is given read from the given file. Can be a gz archive",
    )
    parser.add_argument(
        "--merge",
        help="Merge the two given files, based on the common fields",
        nargs=2,
    )
    parser.add_argument("--stat",
                        help="print statistics about the records file",
                        action="store_true")
    parser.add_argument(
        "--torec",
        help="Convert a column file with given delimiter (see --delimiter) to a rec file. The first line must be the header with the name of the fields like '#field1 #field2 ...'",
        action="store_true",
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func",
                        help="Test only the given function(s)",
                        nargs="+")
    args = parser.parse_args()

    if args.info:
        sys.stdout.write("""\
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


Properties
----------
Properties can be computed from the rec file and added to the output.
See --calc from the help.
Useful properties are implemented:
- num_atoms: compute the number of atoms from a smiles.
             rec --file f.rec --calc 'natoms=num_atoms(smiles)'
             where smiles is a field with a SMILES

        """)
        sys.exit()

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

    if args.sel is not None:
        if os.path.isfile(args.sel):
            with open(args.sel) as f:
                args.sel = f.read().strip()

    if args.torec:
        if args.file is None:
            args.file = sys.stdin
        DATA = columnsfile_to_data(file=args.file, delimiter=args.delimiter)
        dict_to_rec(DATA)
        sys.exit(0)

    if args.find is not None:
        # remove extra spaces:
        findexpr = "("
        nfind = len(args.find)
        for i, find in enumerate(args.find):
            find = re.sub(" +", " ", find)
            argssplit = find.split(" not in ")
            NEGATION = True
            if len(argssplit) == 1:
                argssplit = find.split(" in ")
                NEGATION = False
            assertstr = f"Cannot interpret find expression: {find}"
            assert len(argssplit) == 2, assertstr
            substr = argssplit[0].strip()
            field = argssplit[1].strip()
            if not NEGATION:
                findexpr += f"{field}.find('{substr}')!=-1"
            else:
                findexpr += f"{field}.find('{substr}')==-1"
            if i < nfind - 1:
                if NEGATION:
                    findexpr += " and "
                else:
                    findexpr += " or "
        findexpr += ")"
        if args.sel is None:
            args.sel = findexpr
        else:
            args.sel = f"{args.sel} and {findexpr}"

    if args.merge is not None:
        DATA1 = read_file(
            file=args.merge[0],
            selected_fields=None,
            recsel=args.sel,
            print_records=False,
            print_columns=False,
            rmquote=args.rmquote,
        )
        DATA2 = read_file(
            file=args.merge[1],
            selected_fields=None,
            recsel=args.sel,
            print_records=False,
            print_columns=False,
            rmquote=args.rmquote,
        )
        MERGED = merge_dictionnaries(DATA1, DATA2)
        dict_to_rec(MERGED)
        sys.exit(0)
    if args.file is None:
        args.file = sys.stdin
    if args.calc is not None:
        DATA, _ = get_data(
            file=args.file,
            selected_fields=args.fields,
        )
        calc = re.sub(" +", " ", args.calc)
        name, property = calc.strip().split("=", maxsplit=1)
        DATA = add_property(data=DATA, property=property, name=name)
        dict_to_rec(DATA)
        sys.exit(0)
    if args.run is not None:
        DATA, _ = get_data(file=args.file, rmquote=args.rmquote)
        cmd = re.sub(" +", " ", args.run)
        try:
            names, cmd = cmd.strip().split("=", maxsplit=1)
            names = names.split(",")
        except ValueError:
            sys.exit(
                "check --run argument. The syntax must be 'field==command'. Maybe you forgot to give the output field names ?"
            )
        run(data=DATA, cmd=cmd, fields=args.fields, names=names)
        dict_to_rec(DATA)
        sys.exit(0)
    if args.sort is not None:
        DATA, _ = get_data(file=args.file,
                           selected_fields=args.fields,
                           rmquote=args.rmquote)
        DATA = sort(data=DATA, field=args.sort)
        dict_to_rec(DATA)
        sys.exit(0)
    DATA = read_file(
        file=args.file,
        selected_fields=args.fields,
        recsel=args.sel,
        print_records=args.print_records,
        delimiter=args.delimiter,
        get_stats=args.stat,
        rmquote=args.rmquote,
    )
