#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Feb  7 15:26:27 2024


import gzip
import os

from rdkit import Chem, DataStructs
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from misc import rec


def fpsim(smi1, smi2):
    """
    # FPSim: Fingerprint similarity
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> sim = fpsim(smi1, smi2)
    >>> sim
    0.16223067173637515
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fs1 = Chem.RDKFingerprint(mol1)
    fs2 = Chem.RDKFingerprint(mol2)
    sim = DataStructs.FingerprintSimilarity(fs1, fs2)
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
        sim = fpsim(smi1, smi2)
        out = [smi1, smi2]
        out.extend(info)
        out.append(sim)
        return out

def process_smifile(filename):
    """
    """
    outfilename = os.path.splitext(filename)[0] + "_sim" + ".txt"
    smidataset = SmiDataset(filename)
    smidataloader = DataLoader(smidataset, batch_size=os.cpu_count(), shuffle=False, num_workers=os.cpu_count())
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
        mol = Chem.MolFromSmiles(smi)
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
    >>> len(recdataset)
    1000

    >>> sim  = recdataset[0]
    >>> sim
    0.33902759526938236

    >>> sim = recdataset[1]
    >>> sim
    0.3353096179183136
    """
    def __init__(self, recfilename, key1, key2) -> None:
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.recfilename = recfilename
        self.data, self.fields = rec.get_data(file=recfilename, selected_fields=[key1, key2])

    def __len__(self):
        return len(self.data[self.key1])

    def __getitem__(self, i):
        smi1, smi2 = self.data[self.key1][i], self.data[self.key2][i]
        smi1 = smi1.replace("'", "")
        smi2 = smi2.replace("'", "")
        if isvalid(smi1) and isvalid(smi2):
            sim = fpsim(smi1, smi2)
        else:
            sim = -1
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
    >>> recordsdataset = RecordsDataset(recfilename='molsim_test.rec.gz', similarities=torch.ones(1000))
    >>> sim, record = recordsdataset[3]
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
    sim=1.0
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
                record += f"sim={sim}\n--\n"
                return sim, record
            else:
                record += line + '\n'


def process_recfile(recfile, key1, key2):
    recdataset = RecDataset(recfilename=recfile, key1=key1, key2=key2)
    outfilename = os.path.splitext(recfile)[0] + "_sim" + ".rec.gz"
    recdataloader = DataLoader(recdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=os.cpu_count())
    similarities = list()
    for batch in tqdm(recdataloader, desc="computing similarities"):
        sims = batch
        similarities.extend(list(sims.numpy()))
    recordsdataset = RecordsDataset(recfilename=recfile, similarities=similarities)
    recordsdataloader = DataLoader(recordsdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=1)
    with gzip.open(outfilename, "wt") as outgz:
        for batch in tqdm(recordsdataloader, desc=f"writing file: {outfilename}"):
            sims, records = batch
            for record in records:
                outgz.write(record)


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    parser = argparse.ArgumentParser(description="Compute molecular similarity between the 2 given smiles smi1 and smi2")
    parser.add_argument("--smi1", help='First SMILES string, or rec key if --rec is given', metavar="['Oc1cccc2C(=O)C=CC(=O)c12', 'smi_gen']")
    parser.add_argument("--smi2", help='Second SMILES string, or rech key if --rec is given', metavar="['O1C(=O)C=Cc2cc(OC)c(O)cc12', 'smi_ref']")
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
        sim = fpsim(args.smi1, args.smi2)
        print(f"{sim:.2g}")
    if args.file is not None:
        process_smifile(args.file)
    if args.rec is not None:
        process_recfile(recfile=args.rec, key1=args.smi1, key2=args.smi2)
