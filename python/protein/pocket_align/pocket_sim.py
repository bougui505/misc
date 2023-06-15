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
import os
import pymol2
from tmtools import tm_align
import numpy as np
import scipy.spatial.distance as scidist
from misc.shelve.tempshelve import Tempshelve
import multiprocessing as mp

TEMPSHELVE = Tempshelve()


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def pocket_selector(pdb, lig, radius=6):
    """
    pdb: pdb code or structure filename
    lig: ligand selection in the current pdb or structure filename
    radius: radius, in angstrom,  for the pocket selection

    >>> coords, seq = pocket_selector(pdb="1t4e", lig="resname DIZ", radius=6)
    >>> coords.shape
    (462, 3)
    >>> seq
    'GGGGSSSSSSIIIIIIIIVVVVVVVLLLLLLLLFFFFFFFFFFFLLLLLLLLGGGGQQQQQQQQQIIIIIIIIMMMMMMMMYYYYYYYYYYYYQQQQQQQQQHHHHHHHHHHIIIIIIIIVVVVVVVLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFVVVVVVVKKKKKKKKKEEEEEEEEEHHHHHHHHHHRRRRRRRRRRRIIIIIIIIYYYYYYYYYYYYIIIIIIIIGGGGSSSSSSIIIIIIIIVVVVVVVLLLLLLLLFFFFFFFFFFFLLLLLLLLGGGGQQQQQQQQQIIIIIIIIMMMMMMMMYYYYYYYYYYYYQQQQQQQQQHHHHHHHHHHIIIIIIIIVVVVVVVLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFVVVVVVVKKKKKKKKKEEEEEEEEEHHHHHHHHHHRRRRRRRRRRRIIIIIIIIYYYYYYYYYYYYIIIIIIII'
    """
    key = str((pdb, lig, radius))
    if TEMPSHELVE.has_key(key):
        coords, seq = TEMPSHELVE.get(key)
    else:
        with pymol2.PyMOL() as p:
            p.cmd.set("retain_order", 0)
            p.cmd.set("fetch_path", os.path.expanduser("~/pdb"))
            p.cmd.set("fetch_type_default", "mmtf")
            if os.path.exists(pdb):
                p.cmd.load(filename=pdb, object="myprot")
            else:
                p.cmd.fetch(pdb)
            if os.path.exists(lig):
                p.cmd.load(filename=lig, object="mylig")
            else:
                # print(f"Ligand selection: {lig}")
                p.cmd.select(name="mylig", selection=lig)
            natoms = p.cmd.select(
                name="pocket",
                selection=f"byres(mylig around {radius} and polymer.protein)",
            )
            if natoms > 0:
                coords = p.cmd.get_coords(selection="pocket", state=1)
                myspace = {"seq": []}
                p.cmd.iterate(
                    selection="pocket",
                    expression="seq.append(oneletter)",
                    space=myspace,
                )
                seq = "".join(myspace["seq"])
            else:
                print("Empty pocket. Check the selection!")
                coords = None
                seq = None
        TEMPSHELVE.add(key, (coords, seq))
    return coords, seq


def pocket_sim(
    pdb1=None,
    lig1=None,
    pdb2=None,
    lig2=None,
    radius=6,
    coords1=None,
    seq1=None,
    coords2=None,
    seq2=None,
):
    """
    >>> sim = pocket_sim(pdb1="1t4e_A", lig1="resn DIZ", pdb2="1ycr", lig2="chain B", radius=6)
    >>> sim
    0.7766049848675851
    """
    if coords1 is None or seq1 is None:
        coords1, seq1 = pocket_selector(pdb=pdb1, lig=lig1, radius=radius)
    if coords2 is None or seq2 is None:
        coords2, seq2 = pocket_selector(pdb=pdb2, lig=lig2, radius=radius)
    if coords1 is not None and coords2 is not None:
        res = tm_align(coords1, coords2, seq1, seq2)
        sim = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2
    else:
        sim = None
    return sim


def _read_listfile(listfile):
    """
    >>> _read_listfile(listfile="data/dude_test_16.smi")
    [('data/DUDE100/src/receptor.pdb', 'data/DUDE100/src/crystal_ligand.sdf'), ('data/DUDE100/src/receptor.pdb', ...
    """
    pocketlist = []
    with open(listfile, "r") as f:
        for line in f:
            k, pdb, lig = line.split()
            pocketlist.append((pdb, lig))
    return pocketlist


def _parallel_pocket_sim(args):
    coords1, coords2, seq1, seq2, radius = args
    sim = pocket_sim(
        coords1=coords1, seq1=seq1, coords2=coords2, seq2=seq2, radius=radius
    )
    return sim


def pairwise_pocket_sim(listfile=None, pocketlist=None, radius=6):
    """
    - listfile: text file containing a list formatted as below:
    label pdbfilename ligandfilename

    >>> results, pocketlist = pairwise_pocket_sim("data/dude_test_16.smi")
    >>> results.shape
    (16, 16)
    >>> pocketlist
    [('data/DUDE100/src/receptor.pdb', 'data/DUDE100/src/crystal_ligand.sdf'), ('data/DUDE100/src/receptor.pdb', 'data/DUDE100/src/crystal_ligand.sdf'), ('data/DUDE100/aldr/receptor.pdb', 'data/DUDE100/aldr/crystal_ligand.sdf'), ...

    >>> results, pocketlist = pairwise_pocket_sim(pocketlist=[("data/DUDE100/src/receptor.pdb", "data/DUDE100/src/crystal_ligand.sdf"), ("data/DUDE100/fa10/receptor.pdb", "data/DUDE100/fa10/crystal_ligand.sdf"), ("data/DUDE100/pur2/receptor.pdb", "data/DUDE100/pur2/crystal_ligand.sdf")])
    >>> results
    array([[1.        , 0.44591815, 0.42486217],
           [0.44591815, 1.        , 0.43623708],
           [0.42486217, 0.43623708, 1.        ]])
    """
    p = mp.Pool(processes=os.cpu_count())
    if listfile is not None:
        pocketlist = _read_listfile(listfile=listfile)
    n = len(pocketlist)
    inputs = []
    for i in range(n):
        pdb1, lig1 = pocketlist[i]
        coords1, seq1 = pocket_selector(pdb=pdb1, lig=lig1, radius=radius)
        for j in range(i + 1, n):
            pdb2, lig2 = pocketlist[j]
            coords2, seq2 = pocket_selector(pdb=pdb2, lig=lig2, radius=radius)
            # pocket_sim(pdb1=pdb1, pdb2=pdb2, lig1=lig1, lig2=lig2, radius=radius)
            inputs.append((coords1, coords2, seq1, seq2, radius))
    results = p.map(_parallel_pocket_sim, inputs)
    p.close()
    p.join()
    results = scidist.squareform(np.asarray(results))
    np.fill_diagonal(results, 1.0)
    return results, pocketlist


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

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
    parser.add_argument("-p1", "--pdb1", help="pdbcode or filename")
    parser.add_argument("-p2", "--pdb2", help="pdbcode or filename")
    parser.add_argument(
        "-l1", "--lig1", help="Selection of ligand from pdb1 or filename"
    )
    parser.add_argument(
        "-l2", "--lig2", help="Selection of ligand from pdb2 or filename"
    )
    parser.add_argument(
        "--pairwise",
        help="Pairwise pocket similarity for the given file with format: 'label pdb lig'",
    )
    parser.add_argument(
        "-r",
        "--radius",
        help="radius in angstrom for pocket definition around the ligand",
        type=float,
        default=6.0,
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
    if (
        args.pdb1 is not None
        and args.pdb2 is not None
        and args.lig1 is not None
        and args.lig2 is not None
    ):
        sim = pocket_sim(
            pdb1=args.pdb1,
            pdb2=args.pdb2,
            lig1=args.lig1,
            lig2=args.lig2,
            radius=args.radius,
        )
        print(sim)
    if args.pairwise is not None:
        res, pocketlist = pairwise_pocket_sim(args.pairwise, radius=args.radius)
        n = len(res)
        out = ""
        for i in range(n):
            out += pocketlist[i][0] + " "
            for j in range(n):
                out += f"{res[i,j]:.2f} "
            out += "\n"
        print(out)
