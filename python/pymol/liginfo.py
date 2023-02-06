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
from pymol import cmd
import numpy as np
from misc.protein import coords_loader
from misc import randomgen
from rdkit import Chem
from tqdm import tqdm


def load_pdb(pdb):
    """
    >>> pdb = '1t4e'
    >>> obj = load_pdb(pdb)
    >>> obj # random string
    '...'
    >>> cmd.delete(obj)
    """
    coords, sel = coords_loader.get_coords(pdb, return_selection=True, verbose=False)
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
    resn, resi, chain = identifier.split(':')
    selection = f'{obj} and resn {resn} and resi {resi} and chain {chain}'
    return selection


def lig_to_mol(selection, outmolfilename=None):
    """
    """
    if outmolfilename is None:
        outmolfilename = f'/dev/shm/{randomgen.randomstring()}.mol'
    cmd.save(filename=outmolfilename, selection=selection)
    return outmolfilename


def selection_to_smi(selection):
    """
    """
    molfilename = lig_to_mol(selection)
    mol = Chem.MolFromMolFile(molfilename)
    os.remove(molfilename)
    if mol is None:
        return None
    if mol.GetNumHeavyAtoms() < 5:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
    except:
        sys.stderr.write(f'Cannot convert to SMILES for selection {selection}\n')
        return None
    return smi


def clean_system():
    to_clean = ['polymer.nucleic', 'inorganic', 'solvent', 'metals']
    sel = '|'.join(to_clean)
    cmd.remove(sel)


HEADER = '#SMILES #PDB #resname #chain #resid'


def get_ligands(pdb, delete=True, outsmifilename=None, check_if_protein=True):
    """
    >>> pdb = '1t4e'
    >>> obj, identifiers = get_ligands(pdb, outsmifilename=f'{pdb}.smi')
    >>> identifiers  # fmt: resname:resi:chain
    array(['DIZ:112:A', 'DIZ:112:B'], dtype='<U9')
    """
    obj = load_pdb(pdb)
    clean_system()
    if check_if_protein:
        nres = cmd.select('polymer.protein')
        if nres == 0:
            return None, None
    myspace = {'identifiers': []}
    cmd.iterate(f'{obj} and not polymer.protein', 'identifiers.append(f"{resn}:{resi}:{chain}")', space=myspace)
    identifiers = np.unique(myspace['identifiers'])
    if outsmifilename is not None:
        if not os.path.exists(outsmifilename):
            write_header = True
        else:
            write_header = False
        outsmifile = open(outsmifilename, 'a')
        if write_header:
            outsmifile.write(HEADER + '\n')
    else:
        outsmifile = None
    for identifier in identifiers:
        resn, resi, chain = identifier.split(':')
        selection = identifier_to_selection(identifier, obj)
        smi = selection_to_smi(selection)
        outstr = f'{smi} {pdb} {resn} {chain} {resi}'
        if outsmifile is None:
            print(outstr)
        else:
            outsmifile.write(outstr + '\n')
    if delete:
        cmd.delete(obj)
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


if __name__ == '__main__':
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
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='PDB file(s) or ID(s) to extract the ligand from', nargs='+')
    parser.add_argument('--pdblist', help='Text file with a list of pdbs')
    parser.add_argument('--smi', help='Output SMILES file to write the results out. If not given write to stdout')
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
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    pdblist = []
    if args.pdb is not None:
        pdblist.extend(args.pdb)
    if args.pdblist is not None:
        with open(args.pdblist, 'r') as pdblistfile:
            pdblist.extend([e.strip() for e in pdblistfile.readlines()])
    if args.smi is None:
        print(HEADER)
    else:
        pbar = tqdm(total=len(pdblist))
    for pdb in pdblist:
        get_ligands(pdb, outsmifilename=args.smi)
        if args.smi is not None:
            pbar.update(1)
