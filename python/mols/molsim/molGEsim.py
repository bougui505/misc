#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Feb  7 15:26:27 2024 (adapted for GESim)


import gzip
import os

from misc import rec
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gesim import gesim

RDLogger.DisableLog('rdApp.*')


def gesim_sim(smi1=None, smi2=None, mol1=None, mol2=None):
    """
    # GESim: Graph Entropy Similarity
    >>> smi1 = 'Cc1nn(C2CCN(Cc3cccc(C#N)c3)CC2)cc1-c1ccccc1'
    >>> smi2 = 'Cc1nn(C2CCN(Cc3cccc(Cl)c3)CC2)cc1-c1ccccc1'
    >>> sim = gesim_sim(smi1, smi2)
    >>> sim
    0.9580227325517037
    """
    if mol1 is None:
        mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
    if mol2 is None:
        mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
    sim = gesim.graph_entropy_similarity(mol1, mol2)
    return sim


class SmiDataset(Dataset):
    def __init__(self, filename) -> None:
        super().__init__()
        with open(filename, "r") as f:
            self.lines = f.read().splitlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, index):
        line = self.lines[index]
        data = line.split()
        smi1, smi2 = data[:2]
        info = data[2:]
        sim = gesim_sim(smi1, smi2)
        out = [smi1, smi2]
        out.extend(info)
        out.append(sim)
        return out

def process_smifile(filename):
    """
    """
    outfilename = os.path.splitext(filename)[0] + "_gesim" + ".txt"
    smidataset = SmiDataset(filename)
    smidataloader = DataLoader(smidataset, batch_size=os.cpu_count(), shuffle=False, num_workers=os.cpu_count())  # type: ignore
    with open(outfilename, "w") as outfile:
        for batch in tqdm(smidataloader, "process_smifile"):
            # print(batch)
            smi1_batch = batch[0]
            smi2_batch =  batch[1]
            info_batch = batch[2:-1]
            sim_batch = batch[-1].numpy()
            out = list(zip(smi1_batch, smi2_batch, *info_batch, sim_batch))
            for e in out:
                e = [str(_) for _ in e]
                outfile.write(" ".join(e) + "\n")

def isvalid(smi):
    if smi is not None and len(smi) > 0:
        mol = Chem.MolFromSmiles(smi)  # type: ignore
        if mol is not None:
            valid = True
        else:
            valid = False
    else:
        valid = False
    return valid

class RecDataset(Dataset):
    """
    >>> recdataset = RecDataset(recfilename='molsim_test.rec.gz', key1='smi_gen', key2='smi_ref')
    >>> sim = recdataset[0]
    >>> sim
    0.7766178810633495
    """
    def __init__(self, recfilename, key1, key2, key_mol2_1=None, key_mol2_2=None) -> None:
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.key_mol2_1 = key_mol2_1
        self.key_mol2_2 = key_mol2_2
        self.recfilename = recfilename
        self.data, self.fields = rec.get_data(file=recfilename, selected_fields=[key1, key2, key_mol2_1, key_mol2_2])

    def __len__(self):
        return len(self.data[self.key1])

    def __getitem__(self, i):
        mol1, mol2 = None, None
        if self.key1 is not None:
            smi1 = self.data[self.key1][i]
            smi1 = smi1.replace("'", "")
            mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
        if self.key2 is not None:
            smi2 = self.data[self.key2][i]
            smi2 = smi2.replace("'", "")
            mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
        if self.key_mol2_1 is not None:
            mol2_1 = self.data[self.key_mol2_1][i]
            mol2_1 = mol2_1.replace("'", "")
            mol1 = Chem.rdmolfiles.MolFromMol2File(mol2_1, sanitize=True)  # type: ignore
        if self.key_mol2_2 is not None:
            mol2_2 = self.data[self.key_mol2_2][i]
            mol2_2 = mol2_2.replace("'", "")
            mol2 = Chem.rdmolfiles.MolFromMol2File(mol2_2, sanitize=True)  # type: ignore
        if mol1 is None and mol2 is None:
            return -1
        if mol1 is None:
            return -0.25
        if mol2 is None:
            return -0.5
        sim = -1
        if mol1 is not None and mol2 is not None:  # type: ignore
            sim = gesim_sim(mol1=mol1, mol2=mol2)  # type: ignore
        return sim

def get_len(recfilename):
    """
    >>> get_len(recfilename='molsim_test.rec.gz')
    1000
    """
    n = 0
    with gzip.open(recfilename, "rt") as recfile:
        for line in tqdm(recfile, desc="computing length"):
            line = line.strip()
            if line == '--':
                n += 1
    return n

class RecordsDataset(Dataset):
    """
    >>> import torch
    >>> similarities = torch.ones(1000) * 0.8
    >>> recordsdataset = RecordsDataset(recfilename='molsim_test.rec.gz', similarities=similarities)
    >>> record = recordsdataset[3]
    >>> print(record)
    smi_ref='O=C(O)C1CCN(C(=O)C=Cc2ccc(Sc3ccc4c(c3)OCCO4)c(C(F)(F)F)c2C(F)(F)F)CC1'
    sel='resn_LIG_and_chain_L_and_resi_1'
    pdb='/c7/scratch2/ablondel/GeneMolAI/mkDUDE/all/ital/reclig.pdb'
    smi_gen='CC(=O)N1C(=O)N(c2ccc(Cl)cc2)c2ccccc2C(=O)N(Cc2ccccc2Cl)CC1=O'
    epoch=0
    valid_greedy=0
    label='train'
    valid=1
    smi_gen_greedy='C=C(C)CC1=C(C)C(=O)C(=O)N(Cc1ccccc1)C(=O)NC(Cc1ccc(O)c(O)c1)=NO'
    sim=0.8
    --
    <BLANKLINE>
    """
    def __init__(self, recfilename, similarities) -> None:
        super().__init__()
        self.recfilename = recfilename
        self.recfile = gzip.open(self.recfilename, "rt")
        self.length = get_len(self.recfilename)
        self.similarities = similarities

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        record = ""
        for line in self.recfile:
            line = line.strip()
            if line == '--':
                sim = self.similarities[i]
                record += f"sim={sim}\n"
                record += f"--\n"
                return record
            else:
                record += line + '\n'


def process_recfile(recfile, key1=None, key2=None, key_mol2_1=None, key_mol2_2=None):
    recdataset = RecDataset(recfilename=recfile, key1=key1, key2=key2, key_mol2_1=key_mol2_1, key_mol2_2=key_mol2_2)
    outfilename = os.path.splitext(recfile)[0] + "_gesim" + ".rec.gz"
    recdataloader = DataLoader(recdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=os.cpu_count())  # type: ignore
    similarities = list()
    for sim_batch in tqdm(recdataloader, desc="computing similarities"):
        similarities.extend(list(sim_batch.numpy()))
    recordsdataset = RecordsDataset(recfilename=recfile, similarities=similarities)
    recordsdataloader = DataLoader(recordsdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=1)
    with gzip.open(outfilename, "wt") as outgz:
        for batch in tqdm(recordsdataloader, desc=f"writing file: {outfilename}"):
            records = batch
            for record in records:
                outgz.write(record)


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    parser = argparse.ArgumentParser(description="Compute molecular similarity between the 2 given smiles smi1 and smi2 using Graph Entropy Similarity (GESim)")
    parser.add_argument("--smi1", help='First SMILES string, or rec key if --rec is given', metavar="['Cc1nn(C2CCN(Cc3cccc(C#N)c3)CC2)cc1-c1ccccc1', 'smi_gen']")
    parser.add_argument("--smi2", help='Second SMILES string, or rec key if --rec is given', metavar="['Cc1nn(C2CCN(Cc3cccc(Cl)c3)CC2)cc1-c1ccccc1', 'smi_ref']")
    parser.add_argument("--mol2_1", help='First mol2 file, or rec key if --rec is given')
    parser.add_argument("--mol2_2", help='Second mol2 file, or rec key if --rec is given')
    parser.add_argument("--file", help='Process the given file with the following line format: smi1 smi2 [info1] [...] [infon]. The result will be printed in the last column')
    parser.add_argument("--rec", help='Process the given rec file. The key to read smi1 and smi2 are read from options --smi1 and --smi2 respectively.', metavar='molsim_test.rec.gz')
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

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
    if args.smi1 is not None and args.smi2 is not None and args.rec is None:
        sim = gesim_sim(args.smi1, args.smi2)
        print(f"{sim}")
    if args.file is not None:
        process_smifile(args.file)
    if args.rec is not None:
        process_recfile(recfile=args.rec, key1=args.smi1, key2=args.smi2, key_mol2_1=args.mol2_1, key_mol2_2=args.mol2_2)
