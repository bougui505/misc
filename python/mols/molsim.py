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

class RecDataset(Dataset):
    """
    >>> recdataset = RecDataset(recfilename='molsim_test.rec.gz', key1='smi_gen', key2='smi_ref')
    >>> len(recdataset)
    1000

    >>> sim, record = recdataset[0]
    >>> sim
    0.33902759526938236

    >>> sim, record = recdataset[1]
    >>> sim
    0.3353096179183136
    """
    def __init__(self, recfilename, key1, key2) -> None:
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.recfilename = recfilename
        self.recfile = gzip.open(self.recfilename, "rt")
        self.length = get_len(self.recfilename)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        smi1, smi2 = None, None
        record = ""
        for line in self.recfile:
            line = line.strip()
            if line == '--':
                if isvalid(smi1) and isvalid(smi2):
                    sim = fpsim(smi1, smi2)
                else:
                    sim = -1
                record += f"sim={sim}\n--\n"
                return sim, record
            else:
                record += line + '\n'
                key, val = line.strip().split("=", maxsplit=1)
                if key == self.key1:
                    smi1 = val.replace("'", "")
                if key == self.key2:
                    smi2 = val.replace("'", "")

def process_recfile(recfile, key1, key2):
    recdataset = RecDataset(recfilename=recfile, key1=key1, key2=key2)
    outfilename = os.path.splitext(recfile)[0] + "_sim" + ".rec.gz"
    recdataloader = DataLoader(recdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=1)
    with gzip.open(outfilename, "wt") as outgz:
        for batch in tqdm(recdataloader, desc="computing similarities"):
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
