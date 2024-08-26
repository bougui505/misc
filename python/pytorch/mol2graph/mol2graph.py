#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Aug 23 10:53:18 2024

import os
import subprocess
import tempfile

import torch
from pymol import cmd
from rdkit import Chem
from torch_geometric.data import Data

DISTANCE_BASED = False
D_THRESHOLD = 5.0


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir

def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass

class Mol2:
    """
    >>> mol2 = Mol2()
    >>> mol2.__ATOMTYPES__
    ['C.3', 'C.2', 'C.1', 'C.ar', 'C.cat', 'N.3', 'N.2', 'N.1', 'N.ar', 'N.am', 'N.pl3', 'N.4', 'O.3', 'O.2', 'O.co2', 'O.spc', 'O.t3p', 'S.3', 'S.2', 'S.O', 'S.O2', 'P.3', 'F', 'Cl', 'Br', 'I', 'Sn', 'H', 'H.spc', 'H.t3p', 'LP', 'Du', 'Du.C', 'Any', 'Hal', 'Het', 'Hev', 'Li', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Cr.th', 'Cr.oh', 'Mn', 'Fe', 'Co.oh', 'Cu', 'Zn', 'Se', 'Mo']
    >>> len(mol2.__ATOMTYPES__)
    53
    """
    def __init__(self, distance_based=DISTANCE_BASED, d_threshold=D_THRESHOLD):
        """
        If:
            - distance_based is False, the edge_index are defined based on the BOND section of the mol2 file and the edge_attr are based on the mol2 bond types
            - distance_based is True, the edge_index are defined based on the pairwise distance between atoms and the edge_attr store the distance between atoms below a given distance cutoff (d_threshold).
        """
        self.distance_based = distance_based
        self.d_threshold = d_threshold
        self.atom_id = []
        self.atom_name = []
        self.x = []
        self.y = []
        self.z = []
        self.atom_type = []
        self.bond_id = [] 
        self.origin_atom_id = []
        self.target_atom_id = []
        self.bond_type = []
        self.scriptdir = GetScriptDir()
        self.__ATOMTYPES__ = [l.split()[0] for l in open(f"{self.scriptdir}/mol2.atom_types", "r")]
        self.__BONDTYPES__ = ["1", "2", "3", "am", "ar", "du", "un", "nc"]

    @property
    def coords(self):
        return torch.stack((torch.tensor(self.x), torch.tensor(self.y), torch.tensor(self.z))).T

    @property
    def natoms(self):
        return self.node_features.shape[0]

    @property
    def dmat(self):
        """
        Compute the distance matrix using self.coords
        """
        return torch.cdist(self.coords, self.coords)

    @property
    def edge_index(self):
        if not self.distance_based:
            return torch.stack((torch.tensor(self.origin_atom_id), torch.tensor(self.target_atom_id)))
        else:
            return torch.nonzero(self.dmat < self.d_threshold).T


    @property
    def adjacency(self):
        adjmat = torch.zeros(self.natoms, self.natoms, dtype=torch.int)
        inds = torch.tensor([self.__BONDTYPES__.index(_) for _ in self.bond_type], dtype=torch.int)
        adjmat[self.edge_index[0]-1, self.edge_index[1]-1] = inds
        adjmat[self.edge_index[1]-1, self.edge_index[0]-1] = inds
        return adjmat

    @property
    def node_features(self):
        """
        1-hot encoding of self.atom_type
        """
        inds = [self.__ATOMTYPES__.index(_) for _ in self.atom_type]
        onehot = torch.nn.functional.one_hot(torch.tensor(inds), num_classes=len(self.__ATOMTYPES__))
        return onehot

    @property
    def edge_attr(self):
        """
        1-hot encoding for self.bond_type
        """
        if not self.distance_based:
            inds = [self.__BONDTYPES__.index(_) for _ in self.bond_type]
            onehot = torch.nn.functional.one_hot(torch.tensor(inds), num_classes=len(self.__BONDTYPES__))
            return onehot
        else:
            return self.dmat[*self.edge_index][:, None]


    @property
    def graph(self):
        """
        torch_geometric graph
        """
        g = Data(x=self.node_features, edge_index=self.edge_index, edge_attr=self.edge_attr)
        return g

def graph2mol2(mol2graph):
    """
    Convert a mol2.graph to a mol2
    >>> mol2filename = "data/7zc7_IKL_A_401.mol2"
    >>> mol2 = mol2parser(mol2filename, H=False)
    >>> mol2str = graph2mol2(mol2.graph)
    >>> print(mol2str)
    @<TRIPOS>MOLECULE
    molecule
    26 29
    SMALL
    NO_CHARGES
    @<TRIPOS>ATOM
    1 C 0.0 0.0 0.0 C.2 1 molecule
    2 C 0.0 0.0 0.0 C.2 1 molecule
    3 C 0.0 0.0 0.0 C.2 1 molecule
    ...
    >>> with open("/tmp/out.mol2", "w") as f:
    ...     f.write(mol2str)
    1225
    """
    __ATOMTYPES__ = Mol2().__ATOMTYPES__
    inds = torch.nonzero(mol2graph.x)[:, 1]
    atom_types = [__ATOMTYPES__[_] for _ in inds]
    mol2str = ""
    mol2str += "@<TRIPOS>MOLECULE\n"
    mol2str += "molecule\n"
    mol2str += f"{len(atom_types)} {len(mol2graph.edge_index.T)}\n"
    mol2str += "SMALL\n"
    mol2str += "NO_CHARGES\n"
    mol2str += "@<TRIPOS>ATOM\n"
    atom_id = 1
    for atom_type in atom_types:
        atom_name = atom_type.split(".")[0]
        x = 0.0
        y = 0.0
        z = 0.0
        mol2str += f"{atom_id} {atom_name} {x} {y} {z} {atom_type} 1 molecule\n"
        atom_id+=1
    mol2str += "@<TRIPOS>BOND\n"
    bond_id = 1
    for edge, bond_type in zip(mol2graph.edge_index.T, mol2graph.edge_attr):
        origin_atom_id, target_atom_id = edge
        bond_type = Mol2().__BONDTYPES__[torch.nonzero(bond_type)]
        mol2str += f"{bond_id} {origin_atom_id} {target_atom_id} {bond_type}\n"
        bond_id += 1
    mol2str += "@<TRIPOS>SUBSTRUCTURE\n"
    mol2str += "1 molecule 1"
    return mol2str



def graph2smiles(mol2graph):
    """
    Convert a mol2.graph object to a SMILES
    >>> mol2filename = "data/7zc7_IKL_A_401.mol2"
    >>> mol2 = mol2parser(mol2filename, H=True)
    >>> graph2smiles(mol2.graph)
    'Cc1cc(C)n(-c2ccc(CC(=O)Nc3noc4c3CCCC4)cc2)n1'
    """
    mol2str = graph2mol2(mol2graph)
    mol = Chem.MolFromMol2Block(mol2str)
    smi = Chem.MolToSmiles(mol)
    return smi

def mol2parser(mol2filename, H=True, distance_based=False, d_threshold=D_THRESHOLD):
    """
    For mol2 format description see: https://www.structbio.vanderbilt.edu/archives/amber-archive/2007/att-1568/01-mol2_2pg_113.pdf

    >>> mol2filename = "data/7zc7_IKL_A_401.mol2"
    >>> mol2 = mol2parser(mol2filename, H=False)
    >>> mol2.atom_type
    ['C.2', 'C.2', 'C.2', 'C.3', 'C.3', 'C.3', 'C.2', 'C.3', 'C.2', 'N.2', 'N.3', 'C.2', 'C.3', 'C.2', 'C.2', 'C.2', 'C.2', 'C.3', 'O.2', 'N.3', 'C.2', 'N.2', 'O.3', 'C.2', 'C.3', 'C.2']
    >>> mol2.x
    [-12.743, -12.749, -9.862, -8.78, -7.536, -8.594, -15.381, -17.158, -16.254, -15.933, -15.11, -14.901, -14.059, -15.615, -14.574, -13.242, -13.547, -13.021, -13.579, -11.459, -11.065, -11.74, -10.949, -9.817, -7.741, -14.877]
    >>> mol2.coords
    tensor([[-12.7430, -12.7850,  48.0000],
            [-12.7490, -15.1650,  49.8030],
            [ -9.8620, -16.5570,  47.7560],
            ...
    >>> mol2.bond_id
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> mol2.edge_index.shape
    torch.Size([2, 29])
    >>> mol2.node_features.shape
    torch.Size([26, 50])
    >>> mol2.edge_attr.shape
    torch.Size([29, 8])
    >>> mol2.graph
    Data(x=[26, 50], edge_index=[2, 29], edge_attr=[29, 8])
    >>> mol2.natoms
    26
    >>> mol2.adjacency.shape
    torch.Size([26, 26])

    # Testing with a protein structure using distances between atoms
    >>> mol2filename = "data/reclig.mol2"
    >>> mol2 = mol2parser(mol2filename, H=False, distance_based=True)
    >>> mol2.coords.shape
    torch.Size([2201, 3])
    >>> mol2.dmat.shape
    torch.Size([2201, 2201])
    >>> mol2.graph
    Data(x=[2201, 50], edge_index=[2, 54081], edge_attr=[54081, 1])
    """
    mol2 = Mol2(distance_based=distance_based, d_threshold=d_threshold)
    exclusion = []
    if not H:
        exclusion = ["H", "H.spc", "H.t3p"]
        mol2.__ATOMTYPES__ = [_ for _ in mol2.__ATOMTYPES__ if _  not in exclusion]
    atom_id_dict = dict()
    re_atom_id = 1
    with open(mol2filename, "r") as mol2file:
        section = ""
        H_atom_ids = []
        for line in mol2file:
            line = line.strip()
            if line.startswith("@<TRIPOS>"):
                section = line.split("@<TRIPOS>")[1]
                continue
            if section=="ATOM":
                atom_id, atom_name, x, y, z, atom_type = line.split()[:6]
                atom_id = int(atom_id)
                if not H:
                    if atom_type in exclusion:
                        H_atom_ids.append(atom_id)
                        continue
                atom_id_dict[atom_id] = re_atom_id
                re_atom_id += 1
                mol2.atom_id.append(atom_id_dict[atom_id])
                mol2.atom_name.append(atom_name)
                mol2.x.append(float(x))
                mol2.y.append(float(y))
                mol2.z.append(float(z))
                mol2.atom_type.append(atom_type)
            if section=="BOND":
                bond_id, origin_atom_id, target_atom_id, bond_type = line.split()[:4]
                origin_atom_id, target_atom_id = int(origin_atom_id), int(target_atom_id)
                if origin_atom_id in H_atom_ids or target_atom_id in H_atom_ids:
                    continue
                mol2.bond_id.append(int(bond_id))
                mol2.origin_atom_id.append(atom_id_dict[origin_atom_id])
                mol2.target_atom_id.append(atom_id_dict[target_atom_id])
                mol2.bond_type.append(bond_type)
    return mol2


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
    parser.add_argument("-i", "--inp", help="Input filename to convert to graph")
    parser.add_argument("-s", "--select", help="Optionnal selection")
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
    if args.inp is not None:
        basename, ext = os.path.splitext(args.inp)
        outpt =  basename + '.pt'
        if ext == ".mol2" and args.sel is None:
            mol2 = mol2parser(args.inp)
        else:
            if args.select is None:
                args.select = "all"
            cmd.load(args.inp, object='INPUTFILE')
            pdb_tmp = tempfile.NamedTemporaryFile(suffix='.pdb').name
            mol2_tmp = tempfile.NamedTemporaryFile(suffix='.mol2').name
            cmd.save(pdb_tmp, selection=args.select)
            subprocess.run(f"chimera --nogui {pdb_tmp} {GetScriptDir()}/dockprep.py {mol2_tmp}", shell=True)
            print("dockprep done")
            mol2 = mol2parser(mol2_tmp)
        torch.save(mol2.graph, outpt)
        print(f"{mol2.graph=}")
        print(f"Graph saved in: {outpt}")
