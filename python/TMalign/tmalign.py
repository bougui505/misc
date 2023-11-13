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

import itertools
import os
import subprocess
import tempfile

import pymol2
from joblib import Parallel, delayed
from pymol.creating import gzip
from tqdm import tqdm

from misc import rec


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def clean_states(p):
    objlist_ori = set(p.cmd.get_object_list())
    for obj in objlist_ori:
        p.cmd.split_states(obj)
        p.cmd.delete(obj)
        p.cmd.set_name(f'{obj}_0001', obj)
    objlist = p.cmd.get_object_list()
    for obj in objlist:
        if obj not in objlist_ori:
            p.cmd.delete(obj)


def clean_chains(p, sel, obj, nres_per_chain_min=3):
    chains = p.cmd.get_chains(f"{sel} and {obj}")
    for chain in chains:
        nres = p.cmd.select(
            f"{sel} and polymer.protein and chain {chain} and name CA and present and {obj}")
        if nres <= nres_per_chain_min:
            p.cmd.remove(f"chain {chain} and polymer.protein and {obj}")


def tmalign(model, native, selmodel=None, selnative=None):
    """
    >>> model = "data/1ycr.pdb"
    >>> native = "data/1t4e.pdb"
    >>> tmalign(model, native)
    0.94005

    >>> tmalign(model, native, selmodel='resi 1-30', selnative='resi 1-30')
    0.67607

    Test when less than 3 residues per a chain
    Sequence is too short <3! TMalign error
    >>> model = "data/lt3/5ruv.cif.gz"
    >>> native = "data/lt3/reclig.pdb"
    >>> selmodel = "byres((resn W6J and chain B and resi 201) around 10.0 and polymer.protein)"
    >>> selnative = "byres((resn LIG and chain L and resi 1) around 10.0 and polymer.protein)"
    >>> tmalign(model, native, selmodel=selmodel, selnative=selnative)
    0.1689
    >>> tmalign(native, model, selnative=selmodel, selmodel=selnative)
    0.1689
    """
    selmodel = selmodel if selmodel is not None else "polymer.protein"
    selnative = selnative if selnative is not None else "polymer.protein"
    # model_filename = tempfile.mktemp(suffix=".pdb", dir="/dev/shm")
    # native_filename = tempfile.mktemp(suffix=".pdb", dir="/dev/shm")
    with tempfile.NamedTemporaryFile(
        suffix=".pdb", dir="/dev/shm"
    ) as model_file, tempfile.NamedTemporaryFile(
        suffix=".pdb", dir="/dev/shm"
    ) as native_file:
        pymolsave(model, selmodel, model_file.name)
        pymolsave(native, selnative, native_file.name)
        tmscore = get_tmscore(model_file.name, native_file.name)
    return tmscore


def get_tmscore(modelfile, nativefile):
    scriptdir = GetScriptDir()
    cmd = f"{scriptdir}/USalign {modelfile} {nativefile}".split(
        " ")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True)
    lines = process.stdout.readlines()
    tmscores = []
    for line in lines:
        if line.startswith("TM-score="):
            line = line.strip()
            tmscores.append(float(line.split()[1]))
        if len(tmscores) == 2:
            break
    tmscore = max(tmscores)
    return tmscore


def pymolsave(infile, sel, outfile):
    with pymol2.PyMOL() as p:
        p.cmd.feedback(action='disable', module='all', mask='everything')
        p.cmd.load(filename=infile, object="mymodel")
        # Remove alternate locations
        p.cmd.remove("not alt ''+A")
        p.cmd.alter("all", "alt=''")
        ############################
        clean_states(p)
        clean_chains(p, f"mymodel and ({sel})", obj="mymodel")
        p.cmd.save(filename=outfile,
                   selection=f"mymodel and ({sel})", state=-1)


def tmalign_wrapper(model, native, selmodel, selnative):
    try:
        tmscore = tmalign(model=model, native=native,
                          selmodel=selmodel, selnative=selnative)
    except Exception as e:
        print("ERROR for", model, native, selmodel, selnative, file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        tmscore = -1.0
    return tmscore


def tmalign_multi(model_list, native_list, selmodel_list=None, selnative_list=None, verbose=False):
    """
    Align pairwisely the model_list and the native_list
    """
    if os.path.exists("tmalign.err.gz"):
        os.remove("tmalign.err.gz")
    if selmodel_list is None:
        selmodel_list = [None]*len(model_list)
    assert len(model_list) == len(selmodel_list)

    if selnative_list is None:
        selnative_list = [None]*len(native_list)
    assert len(native_list) == len(selnative_list)

    n_models = len(model_list)
    n_natives = len(native_list)

    ncpu = os.cpu_count()
    iterproduct = itertools.product(zip(model_list, selmodel_list),
                                    zip(native_list, selnative_list))
    # for ((model, selmodel), (native, selnative)) in iterproduct:
    #     print(model, selmodel, native, selnative)
    tmscores = Parallel(n_jobs=ncpu)(delayed(tmalign_wrapper)(model=model, native=native, selmodel=selmodel,
                                                              selnative=selnative) for ((model, selmodel), (native, selnative)) in tqdm(iterproduct, total=n_models*n_natives, ncols=64, position=1, leave=False))
    iterproduct = zip(itertools.product(zip(model_list, selmodel_list),
                                        zip(native_list, selnative_list)), tmscores)
    if verbose:
        for (((model, selmodel), (native, selnative)), tmscore) in iterproduct:
            print(f"{model=}")
            print(f"{selmodel=}")
            print(f"{native=}")
            print(f"{selnative=}")
            print(f"{tmscore=}")
            print("--")
    return tmscores


def tmalign_pairwise(pdb_list, sel_list, outfilename):
    n = len(pdb_list)
    if sel_list is None:
        sel_list = [None,] * n
    with gzip.open(outfilename, 'wt') as gz:
        for i, (prot1, prot1_sel) in tqdm(enumerate(zip(pdb_list, sel_list)), total=len(pdb_list), ncols=64, position=0):
            tmscores = tmalign_multi(pdb_list[i:], [prot1],
                                     selmodel_list=sel_list[i:], selnative_list=[prot1_sel], verbose=False)
            assert len(tmscores) == len(pdb_list[i:])
            assert len(tmscores) == len(sel_list[i:])
            for j, (tmscore, prot2, prot2_sel) in enumerate(zip(tmscores, pdb_list[i:], sel_list[i:])):
                j = j + i
                if tmscore >= 0.0:
                    distance = 1.0 - tmscore
                else:
                    distance = -1.0
                gz.write(f"{prot1=}\n")
                gz.write(f"{prot1_sel=}\n")
                gz.write(f"{i=}\n")
                gz.write(f"{prot2=}\n")
                gz.write(f"{prot2_sel=}\n")
                gz.write(f"{j=}\n")
                gz.write(f"{tmscore=}\n")
                gz.write(f"{distance=}\n")
                gz.write("--\n")


def tmalign_rec(recfile):
    outfile = recfile.split(".")[0] + '_tmscore.rec.gz'
    assert not os.path.exists(
        outfile), f"{outfile} already exists, please remove it"
    data, fields = rec.get_data(file=recfile, rmquote=True)
    assert 'pdb1' in fields
    assert 'pdb2' in fields
    pdb1list = data["pdb1"]
    pdb2list = data["pdb2"]
    n = len(pdb1list)
    if "sel1" in fields:
        sel1list = data["sel1"]
    else:
        sel1list = [None,] * n
    if "sel2" in fields:
        sel2list = data["sel2"]
    else:
        sel2list = [None,] * n
    ncpu = os.cpu_count()
    tmscores = Parallel(n_jobs=ncpu)(delayed(tmalign_wrapper)(model=pdb1, native=pdb2, selmodel=sel1, selnative=sel2) for (
        pdb1, pdb2, sel1, sel2) in tqdm(zip(pdb1list, pdb2list, sel1list, sel2list), total=n, ncols=64, position=1, leave=False))
    assert len(tmscores) == n
    with gzip.open(outfile, 'wt') as gz:
        for i in range(n):
            for field in fields:
                gz.write(f"{field}={data[field][i]}\n")
            gz.write(f"tmscore={tmscores[i]}\n")
            gz.write(f"distance={1.0 - tmscores[i]}\n")
            gz.write("--\n")


def read_csv(csvfilename):
    pathlist = []
    sellist = []
    with open(csvfilename, "r") as csvfile:
        for line in csvfile:
            splitted = line.strip().split(",")
            if len(splitted) >= 2:
                path, sel = splitted[:2]
            else:
                path = splitted[0]
                sel = None
            pathlist.append(path)
            sellist.append(sel)
    return pathlist, sellist


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
    parser.add_argument("-m", "--model")
    parser.add_argument("-n", "--native")
    parser.add_argument("--selmodel")
    parser.add_argument("--selnative")
    parser.add_argument(
        "--model_list", help="CSV file with list of models. Col1: path to structure files; Col2 (optionnal): selection for the given file")
    parser.add_argument(
        "--native_list", help="CSV file with list of natives. Col1: path to structure files; Col2 (optionnal): selection for the given file")
    parser.add_argument(
        "--pairwise", help="Pairwise alignment for the given CSV file. Col1: path to structure files; Col2 (optionnal): selection for the given file")
    parser.add_argument(
        "--rec", help="Compute TMscores from the given rec file. The fields must be: pdb1, sel1 (optionnal), pdb2, sel2 (optionnal). All other fields are also kept in the output")
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
    if args.model is not None and args.native is not None:
        tmscore = tmalign(model=args.model, native=args.native,
                          selmodel=args.selmodel, selnative=args.selnative)
        print(f"{tmscore=}")
    if args.model_list is not None and args.native_list is not None:
        model_list, selmodel_list = read_csv(args.model_list)
        native_list, selnative_list = read_csv(args.native_list)
        tmalign_multi(model_list=model_list, native_list=native_list,
                      selmodel_list=selmodel_list, selnative_list=selnative_list, verbose=True)
    if args.pairwise is not None:
        pdb_list, sel_list = read_csv(args.pairwise)
        outfilename = os.path.splitext(args.pairwise)[
            0] + "_pairwise_tmscore.rec.gz"
        if os.path.exists(outfilename):
            sys.exit(f"{outfilename} already exists")
        tmalign_pairwise(pdb_list=pdb_list, sel_list=sel_list,
                         outfilename=outfilename)
    if args.rec is not None:
        tmalign_rec(args.rec)
