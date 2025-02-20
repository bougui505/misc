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

from misc import rec
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdRascalMCES
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import MCS_similarity

RDLogger.DisableLog('rdApp.*')


def fpsim(smi1=None, smi2=None, mol1=None, mol2=None):
    """
    # FPSim: Fingerprint similarity
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> sim = fpsim(smi1, smi2)
    >>> sim
    0.16223067173637515
    """
    if mol1 is None:
        mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
    if mol2 is None:
        mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
    fs1 = Chem.RDKFingerprint(mol1)  # type: ignore
    fs2 = Chem.RDKFingerprint(mol2)  # type: ignore
    sim = DataStructs.FingerprintSimilarity(fs1, fs2)
    return sim

def mcs_sim(smi1=None, smi2=None, mol1=None, mol2=None):
    """
    mcs_sim: Maximum common substructure similarity
    See: https://github.com/shuan4638/mcs_sim/blob/88f71fa6795101bcfd7a78a33652f0366077ce16/MCS_similarity.py#L57
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> sim = mcs_sim(smi1, smi2)
    >>> sim
    0.5185185185185185
    """
    if mol1 is None:
        mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
    if mol2 is None:
        mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
    res = rdFMCS.FindMCS([mol1, mol2], ringMatchesRingOnly=True,completeRingsOnly=True)
    return (2*res.numAtoms)/(mol1.GetNumAtoms()+mol2.GetNumAtoms())

def rascal_sim(smi1=None, smi2=None, mol1=None, mol2=None):
    """
    See: https://greglandrum.github.io/rdkit-blog/posts/2023-11-08-introducingrascalmces.html#similarity-threshold
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> rascal_sim(smi1=smi1, smi2=smi2)
    -1.0
    """
    if mol1 is None:
        mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
    if mol2 is None:
        mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.7
    # opts.timeout = 1
    # opts.returnEmptyMCES = True
    results = rdRascalMCES.FindMCES(mol1, mol2, opts)
    # print(results[0].tier1Sim, results[0].tier2Sim)
    try:
        return results[0].similarity
    except IndexError:
        return -1.0

def murcko_sim(smi1=None, smi2=None, mol1=None, mol2=None):
    """
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> murcko_sim(smi1=smi1, smi2=smi2)
    0.9629629629629629
    """
    if mol1 is None:
        mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
    if mol2 is None:
        mol2 = Chem.MolFromSmiles(smi2)  # type: ignore
    # core1 = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol1))
    # core2 = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol2))
    core1 = MurckoScaffold.GetScaffoldForMol(mol1)
    core2 = MurckoScaffold.GetScaffoldForMol(mol2)
    fs1 = Chem.RDKFingerprint(core1)  # type: ignore
    fs2 = Chem.RDKFingerprint(core2)  # type: ignore
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
    >>> recdataset = RecDataset(recfilename='molsim_test.rec.gz', key1='smi_gen', key2='smi_ref', mcs=True, fastmcs=True, rascal=True, murcko=True)
    >>> sim, sim_mcs, sim_rascal, sim_murcko  = recdataset[0]
    >>> sim
    0.33902759526938236
    >>> sim_mcs
    0.1111111111111111
    >>> sim_rascal
    -1.0
    >>> sim_murcko
    0.9629629629629629
    """
    def __init__(self, recfilename, key1, key2, key_mol2_1=None, key_mol2_2=None, mcs=False, fastmcs=False, rascal=False, murcko=False) -> None:
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.key_mol2_1 = key_mol2_1
        self.key_mol2_2 = key_mol2_2
        self.recfilename = recfilename
        self.data, self.fields = rec.get_data(file=recfilename, selected_fields=[key1, key2, key_mol2_1, key_mol2_2])
        self.mcs = mcs
        self.fastmcs = fastmcs
        self.rascal = rascal
        self.murcko = murcko

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
            # if mol1 is None:
            #     mol1 = Chem.rdmolfiles.MolFromMol2File(mol2_1, sanitize=False)  # type: ignore
        if self.key_mol2_2 is not None:
            mol2_2 = self.data[self.key_mol2_2][i]
            mol2_2 = mol2_2.replace("'", "")
            mol2 = Chem.rdmolfiles.MolFromMol2File(mol2_2, sanitize=True)  # type: ignore
            # if mol2 is None:
            #     mol2 = Chem.rdmolfiles.MolFromMol2File(mol2_2, sanitize=False)  # type: ignore
        if mol1 is None and mol2 is None:
            return -1, -1, -1, -1
        if mol1 is None:
            return -0.25, -0.25, -0.25, -0.25
        if mol2 is None:
            return -0.5, -0.5, -0.5, -0.5
        sim, sim_mcs, sim_rascal, sim_murcko = -1, -1, -1, -1
        if mol1 is not None and mol2 is not None:  # type: ignore
            sim = fpsim(mol1=mol1, mol2=mol2)  # type: ignore
            if self.mcs or self.fastmcs:
                if self.fastmcs:
                    sim_mcs = MCS_similarity.fast_MCS_Sim(mol1=mol1, mol2=mol2)
                else:
                    sim_mcs = mcs_sim(mol1=mol1, mol2=mol2)
            else:
                sim_mcs = -1
            if self.rascal:
                sim_rascal = rascal_sim(mol1=mol1, mol2=mol2)
            if self.murcko:
                sim_murcko = murcko_sim(mol1=mol1, mol2=mol2)
        return sim, sim_mcs, sim_rascal, sim_murcko

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
    >>> recordsdataset = RecordsDataset(recfilename='molsim_test.rec.gz', similarities=torch.ones(1000), similarities_mcs=torch.ones(1000)*2, similarities_rascal=torch.ones(1000)*3, similarities_murcko=torch.ones(1000)*4)
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
    sim=1.0
    sim_mcs=2.0
    sim_murcko=4.0
    sim_rascal=3.0
    --
    <BLANKLINE>
    """
    def __init__(self, recfilename, similarities, similarities_mcs, similarities_rascal, similarities_murcko) -> None:
        super().__init__()
        self.recfilename = recfilename
        self.recfile = gzip.open(self.recfilename, "rt")
        self.length = get_len(self.recfilename)
        self.similarities = similarities
        self.similarities_mcs = similarities_mcs
        self.similarities_rascal = similarities_rascal
        self.similarities_murcko = similarities_murcko

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        record = ""
        for line in self.recfile:
            line = line.strip()
            if line == '--':
                sim = self.similarities[i]
                sim_mcs = self.similarities_mcs[i]
                sim_rascal =  self.similarities_rascal[i]
                sim_murcko = self.similarities_murcko[i]
                record += f"sim={sim}\n"
                record += f"sim_mcs={sim_mcs}\n"
                record += f"sim_murcko={sim_murcko}\n"
                record += f"sim_rascal={sim_rascal}\n--\n"
                return record
            else:
                record += line + '\n'


def process_recfile(recfile, key1=None, key2=None, key_mol2_1=None, key_mol2_2=None, mcs=False, fastmcs=False, rascal=False, murcko=False):
    recdataset = RecDataset(recfilename=recfile, key1=key1, key2=key2, key_mol2_1=key_mol2_1, key_mol2_2=key_mol2_2, mcs=mcs, fastmcs=fastmcs, rascal=rascal, murcko=murcko)
    outfilename = os.path.splitext(recfile)[0] + "_sim" + ".rec.gz"
    recdataloader = DataLoader(recdataset, batch_size=os.cpu_count(), shuffle=False, num_workers=os.cpu_count())  # type: ignore
    similarities = list()
    similarities_mcs = list()
    similarities_rascal = list()
    similarities_murcko = list()
    for batch in tqdm(recdataloader, desc="computing similarities"):
        sims, sims_mcs, sims_rascal, sims_murcko = batch
        similarities.extend(list(sims.numpy()))
        similarities_mcs.extend(list(sims_mcs.numpy()))
        similarities_rascal.extend(list(sims_rascal.numpy()))
        similarities_murcko.extend(list(sims_murcko.numpy()))
    recordsdataset = RecordsDataset(recfilename=recfile, similarities=similarities, similarities_mcs=similarities_mcs, similarities_rascal=similarities_rascal, similarities_murcko=similarities_murcko)
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

    parser = argparse.ArgumentParser(description="Compute molecular similarity between the 2 given smiles smi1 and smi2")
    parser.add_argument("--smi1", help='First SMILES string, or rec key if --rec is given', metavar="['Oc1cccc2C(=O)C=CC(=O)c12', 'smi_gen']")
    parser.add_argument("--smi2", help='Second SMILES string, or rech key if --rec is given', metavar="['O1C(=O)C=Cc2cc(OC)c(O)cc12', 'smi_ref']")
    parser.add_argument("--mol2_1", help='First mol2 file, or rec key if --rec is given')
    parser.add_argument("--mol2_2", help='Second mol2 file, or rec key if --rec is given')
    parser.add_argument("--file", help='Process the given file with the following line format: smi1 smi2 [info1] [...] [infon]. The result will be printed in the last column')
    parser.add_argument("--rec", help='Process the given rec file. The key to read smi1 and smi2 are read from options --smi1 and --smi2 respectively.', metavar='molsim_test.rec.gz')
    parser.add_argument("--mcs", help="Compute Maximum Common Substructure similarity (sim_mcs)", action="store_true")
    parser.add_argument("--fastmcs", help="Compute a fast approximation of the Maximum Common Substructure similarity (sim_mcs)", action="store_true")
    parser.add_argument("--rascal", help="Compute rascal similarity (see: https://greglandrum.github.io/rdkit-blog/posts/2023-11-08-introducingrascalmces.html#similarity-threshold)", action="store_true")
    parser.add_argument("--murcko", help="Compute similarity between Murcko scaffold", action="store_true")
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
        process_recfile(recfile=args.rec, key1=args.smi1, key2=args.smi2, key_mol2_1=args.mol2_1, key_mol2_2=args.mol2_2, mcs=args.mcs, fastmcs=args.fastmcs, rascal=args.rascal, murcko=args.murcko)
