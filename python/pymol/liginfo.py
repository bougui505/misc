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

import numpy as np
from misc.protein import coords_loader
from pymol import cmd
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from tqdm import tqdm

from misc import randomgen


def load_pdb(pdb, fetch_path=os.path.expanduser("~/pdb")):
    """
    >>> pdb = '1t4e'
    >>> obj = load_pdb(pdb)
    >>> obj # random string
    '...'
    >>> cmd.delete(obj)
    """
    coords, sel = coords_loader.get_coords(
        pdb, return_selection=True, verbose=False, fetch_path=fetch_path
    )
    obj = sel.split()[0]
    return obj


def identifier_to_selection(identifier, obj):
    """
    >>> identifiers = ['DIZ:112:A', 'DIZ:112:B']
    >>> obj = 'obj'
    >>> identifier_to_selection(identifiers[0], obj)
    'obj and resn DIZ and resi 112 and chain A'
    """
    selections = []
    resn, resi, chain = identifier.split(":")
    selection = f"{obj} and resn {resn} and resi {resi} and chain {chain}"
    return selection


def lig_to_mol(selection, outmolfilename=None):
    """ """
    if outmolfilename is None:
        outmolfilename = f"/dev/shm/{randomgen.randomstring()}.mol"
    cmd.save(filename=outmolfilename, selection=selection)
    return outmolfilename


def selection_to_smi(selection, sanitize=True):
    """ """
    molfilename = lig_to_mol(selection)
    mol = Chem.MolFromMolFile(molfilename, sanitize=sanitize)
    os.remove(molfilename)
    if mol is None:
        return "-"
    try:
        smi = Chem.MolToSmiles(mol)
    except:
        sys.stderr.write(
            f"Cannot convert to SMILES for selection {selection}\n")
        return "-"
    if len(smi) == 0:
        return "-"
    return smi


def clean_system():
    to_clean = ["polymer.nucleic", "inorganic", "solvent"]  # , 'metals'
    sel = "|".join(to_clean)
    cmd.remove(sel)
    # Remove alternate locations (see: https://pymol.org/dokuwiki/doku.php?id=concept:alt):
    cmd.remove("not alt ''+A")
    cmd.alter("all", "alt=''")


HEADER = "#SMILES #PDB #resname #chain #resid"


def compute_tanimoto(smi1, smi2):
    """
    >>> smi1 = "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O"
    >>> smi2 = "N=C(N)c1ccc(CC(O)C(=O)O)cc1"
    >>> compute_tanimoto(smi1, smi2)
    0.32151898734177214
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)
    s = DataStructs.TanimotoSimilarity(fp1, fp2)
    return s


def get_ligands(
    pdb,
    delete=True,
    outsmifilename=None,
    check_if_protein=True,
    sanitize=True,
    smi_ref=None,
    fetch_path=os.path.expanduser("~/pdb"),
    user_selection="all",
    no_smi=False
):
    """
    - if no_smi is True, do not compute smiles
    >>> pdb = '1t4e'
    >>> obj, identifiers = get_ligands(pdb, outsmifilename=f'{pdb}.smi')
    >>> identifiers  # fmt: resname:resi:chain
    array(['DIZ:112:A', 'DIZ:112:B'], dtype='<U9')

    # If smi_ref is not None compute the Tanimoto similarity with the reference and returns it in last column
    >>> obj, identifiers = get_ligands(pdb, smi_ref="N=C(N)c1ccc(CC(O)C(=O)O)cc1")
    O=C(O)[C@H](c1ccc(Cl)cc1)N1C(=O)c2cc(I)ccc2NC(=O)[C@@H]1c1ccc(Cl)cc1 1t4e DIZ A 112 0.4542
    O=C(O)[C@H](c1ccc(Cl)cc1)N1C(=O)c2cc(I)ccc2NC(=O)[C@@H]1c1ccc(Cl)cc1 1t4e DIZ B 112 0.4542
    """
    obj = load_pdb(pdb, fetch_path=fetch_path)
    clean_system()
    if check_if_protein:
        nres = cmd.select("polymer.protein")
        if nres == 0:
            return None, None
    cmd.remove("polymer.protein")
    cmd.remove(f"not ({user_selection})")
    myspace = {"identifiers": []}
    cmd.iterate(
        f"{obj} and not polymer.protein",
        'identifiers.append(f"{resn}:{resi}:{chain}")',
        space=myspace,
    )
    identifiers = np.unique(myspace["identifiers"])
    if outsmifilename is not None:
        if not os.path.exists(outsmifilename):
            write_header = True
        else:
            write_header = False
        outsmifile = open(outsmifilename, "a")
        if write_header:
            if smi_ref is None:
                outsmifile.write(HEADER + "\n")
            else:
                outsmifile.write(HEADER + " #tanimoto" + "\n")
    else:
        outsmifile = None
    for identifier in identifiers:
        resn, resi, chain = identifier.split(":")
        selection = identifier_to_selection(identifier, obj)
        selection = f"{selection} and {user_selection}"
        if cmd.select(selection) == 0:  # empty selection
            continue
        if no_smi:
            smi = "-"
        else:
            smi = selection_to_smi(selection, sanitize=sanitize)
        outstr = f"{smi} {pdb} {resn} {chain} {resi}"
        if (
            smi is None
            or smi == "error_MolFromMolFile"
            or smi == "error_MolToSmiles"
            or smi == "empty_smi"
        ):
            outstr = "#" + outstr
        else:
            if smi_ref is not None:
                tanimoto = compute_tanimoto(smi, smi_ref)
                outstr += f" {tanimoto:.4g}"
        if outsmifile is None:
            print(outstr)
        else:
            outsmifile.write(outstr + "\n")
    if delete:
        cmd.delete("all")
    return obj, identifiers


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
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "--pdb", help="PDB file(s) or ID(s) to extract the ligand from", nargs="+"
    )
    parser.add_argument(
        "--pdblist",
        help="Text file with a list of pdbs. If a smi is given in the second column, compute the Tanimoto similarity with the reference smiles",
    )
    parser.add_argument("--sel", help="selection", default="all")
    parser.add_argument(
        "--ref", help="Reference SMILES to compute Tanimoto with")
    parser.add_argument(
        "--smi",
        help="Output SMILES file to write the results out. If not given write to stdout",
    )
    parser.add_argument(
        "--no_sanitize",
        help="Do not sanitize the molecule",
        dest="sanitize",
        action="store_false",
    )
    parser.add_argument(
        "--no_smi", help="Do not compute SMILES", action="store_true")
    parser.add_argument(
        "--fetch_path",
        help="Directory where to store the pdb files (default: ~/pdb)",
        default=os.path.expanduser("~/pdb"),
    )
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

    pdblist = []
    smireflist = []
    if args.pdb is not None:
        pdblist.extend(args.pdb)
    if args.ref is not None:
        smireflist.append(args.ref)
    if args.pdblist is not None:
        with open(args.pdblist, "r") as pdblistfile:
            for line in pdblistfile.readlines():
                line = line.strip().split()
                pdblist.append(line[0])
                if len(line) == 2:
                    smireflist.append(line[1])
        if len(smireflist) > 0:
            assert len(pdblist) == len(smireflist)
    if args.smi is None:
        print(HEADER)
    else:
        pbar = tqdm(total=len(pdblist))
    if len(smireflist) > 0:
        toiter = zip(pdblist, smireflist)
        is_ref = True
    else:
        toiter = pdblist
        is_ref = False
    for elem in toiter:
        if is_ref:
            pdb, smi_ref = elem
        else:
            pdb = elem
            smi_ref = None
        if not args.sanitize:
            print("# not-sanitized molecules")
        get_ligands(
            pdb,
            outsmifilename=args.smi,
            sanitize=args.sanitize,
            smi_ref=smi_ref,
            fetch_path=args.fetch_path,
            user_selection=args.sel,
            no_smi=args.no_smi
        )
        if args.smi is not None:
            pbar.update(1)
