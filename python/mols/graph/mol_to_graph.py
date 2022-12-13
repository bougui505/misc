#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
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
import os
import torch
import torch.nn.functional as F
from misc.mols.rdkit_fix import molfromsmiles
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.data import Dataset

allowable_features = {
    'possible_atomic_num_list':
    list(range(1, 119)) + ['misc'],
    'possible_chirality_list': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM',
        'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'
    ],
    'possible_atom_type_2': [
        'C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD', 'OE', 'OG', 'OH',
        'OX', 'S*', 'SD', 'SG', 'misc'
    ],
    'possible_atom_type_3': [
        'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3',
        'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
        'OXT', 'SD', 'SG', 'misc'
    ],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def lig_atom_featurizer(mol):
    """
    >>> smiles = "O[C@@H]([C@H]1O[C@H]([C@H](O)[C@@H]1O)n1ccc2C3=NCC(O)N3C=Nc12)c1ccc(Cl)cc1"
    >>> mol = molfromsmiles(smiles)
    >>> atom_features = lig_atom_featurizer(mol)
    >>> mol.GetNumAtoms()
    48
    >>> atom_features.shape
    torch.Size([48, 16])
    """
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])
    return torch.tensor(atom_features_list, dtype=torch.float)


def get_mol_graph(mol):
    """
    Adapted from: get_lig_graph from diffdock (https://github.com/gcorso/DiffDock)
    See: https://github.com/gcorso/DiffDock/blob/8e853d6b14fb57baf90fa8529349117439f06819/datasets/process_mols.py#L248

    >>> smiles = "O[C@@H]([C@H]1O[C@H]([C@H](O)[C@@H]1O)n1ccc2C3=NCC(O)N3C=Nc12)c1ccc(Cl)cc1"
    >>> mol = molfromsmiles(smiles)
    >>> graph = get_mol_graph(mol)
    >>> graph
    Data(x=[48, 16], edge_index=[2, 104], edge_attr=[104, 4], pos=[48, 3], edge_type=[104])
    """
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
    data = Data(x=atom_feats, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, pos=lig_coords)
    return data


class MolDataset(Dataset):
    """
    >>> smilesfilename = 'data/test.smi'
    >>> dataset = MolDataset(smilesfilename)
    >>> dataset.get(10)
    Data(x=[52, 16], edge_index=[2, 110], edge_attr=[110, 4], pos=[52, 3], edge_type=[110])

    >>> from torch_geometric.loader import DataLoader
    >>> seed = torch.manual_seed(0)
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())
    >>> iterator = iter(loader)
    >>> next(iterator)
    DataBatch(x=[1867, 16], edge_index=[2, 3912], edge_attr=[3912, 4], pos=[1867, 3], edge_type=[3912], batch=[1867], ptr=[33])

    >>> smilesfilename = 'data/HMT_mols_test.smi'
    >>> dataset = MolDataset(smilesfilename, readclass=True)
    >>> graph, graphclass = dataset.get(10)
    >>> graph
    Data(x=[48, 16], edge_index=[2, 106], edge_attr=[106, 4], pos=[48, 3], edge_type=[106])
    >>> graphclass
    'CARM1'
    >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count())
    >>> iterator = iter(loader)
    >>> graphs, classes = next(iterator)
    >>> graphs
    DataBatch(x=[1727, 16], edge_index=[2, 3646], edge_attr=[3646, 4], pos=[1727, 3], edge_type=[3646], batch=[1727], ptr=[33])
    >>> classes
    ('PRMT1', 'SMYD2', 'PRDM6', 'CARM1', 'PRMT5', 'PRMT2', 'PRMT1', 'FBL', 'PRMT1', 'PRMT2', 'PCMT1', 'PRMT1', 'PRMT1', 'PRMT2', 'CARM1', 'MLL', 'MLL', 'PRDM6', 'PCMT1', 'CARM1', 'SETD2', 'CARM1', 'PRMT5', 'PRMT1', 'PRMT1', 'CARM1', 'PRDM16', 'PRDM5', 'PCMT1', 'PRMT2', 'PRMT5', 'MLL')
    """
    def __init__(self, smilesfilename, readclass=False):
        super().__init__()
        self.readclass = readclass
        self.smilesfilename = smilesfilename

    def len(self):
        smilesfile = open(self.smilesfilename, 'r')
        n = sum(1 for line in smilesfile)
        return n

    def get_line(self, infile, index):
        for i, line in enumerate(infile):
            if i == index:
                break
        infile.seek(0)
        return line.strip()

    def get(self, idx):
        smilesfile = open(self.smilesfilename, 'r')
        inline = self.get_line(smilesfile, idx).split()
        smiles = inline[0]
        if self.readclass:
            smiles_class = inline[1]
        mol = molfromsmiles(smiles)
        graph = get_mol_graph(mol)
        if self.readclass:
            return graph, smiles_class
        else:
            return graph


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
    parser.add_argument('-a', '--arg1')
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
