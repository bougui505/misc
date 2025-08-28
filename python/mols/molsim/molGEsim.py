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
import multiprocessing
import numpy as np

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
    try:
        if mol1 is None and smi1 is not None:
            mol1 = Chem.MolFromSmiles(smi1)  # type: ignore
        if mol2 is None and smi2 is not None:
            mol2 = Chem.MolFromSmiles(smi2)  # type: ignore

        if mol1 is None or mol2 is None:
            return float('nan') # Return NaN if one of the molecules couldn't be parsed from smiles/mol2

        sim = gesim.graph_entropy_similarity(mol1, mol2)
        return sim
    except Exception:
        # Catch any other Python exceptions during similarity computation
        return float('nan') # Return NaN on any Python exception


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
        smi1_val, smi2_val = None, None
        mol2_1_path_val, mol2_2_path_val = None, None

        # Extract raw string values for smiles and mol2 paths
        if self.key1 is not None:
            smi1_val = self.data[self.key1][i].replace("'", "")
        if self.key2 is not None:
            smi2_val = self.data[self.key2][i].replace("'", "")
        if self.key_mol2_1 is not None:
            mol2_1_path_val = self.data[self.key_mol2_1][i].replace("'", "")
        if self.key_mol2_2 is not None:
            mol2_2_path_val = self.data[self.key_mol2_2][i].replace("'", "")

        # Use _compute_single_record_sim's logic for parsing and robustness
        sim = _compute_single_record_sim((smi1_val, smi2_val, mol2_1_path_val, mol2_2_path_val))
        return sim

def _compute_single_record_sim(args):
    """
    Helper function to compute similarity for a single record.
    This function runs in a separate process.
    """
    smi1, smi2, mol2_1_path, mol2_2_path = args
    mol1, mol2 = None, None

    # Parse mol1 from smi1 or mol2_1_path
    if smi1 is not None:
        try:
            mol1 = Chem.MolFromSmiles(smi1)
        except Exception:
            pass
    elif mol2_1_path is not None:
        try:
            mol1 = Chem.rdmolfiles.MolFromMol2File(mol2_1_path, sanitize=True)
        except Exception:
            pass

    # Parse mol2 from smi2 or mol2_2_path
    if smi2 is not None:
        try:
            mol2 = Chem.MolFromSmiles(smi2)
        except Exception:
            pass
    elif mol2_2_path is not None:
        try:
            mol2 = Chem.rdmolfiles.MolFromMol2File(mol2_2_path, sanitize=True)
        except Exception:
            pass

    # Call the robust gesim_sim
    # gesim_sim itself handles if mol1 or mol2 are None and any further Python exceptions.
    return gesim_sim(mol1=mol1, mol2=mol2, smi1=smi1, smi2=smi2) # Pass mol objects and original smiles


def process_recfile(recfile, key1=None, key2=None, key_mol2_1=None, key_mol2_2=None):
    # Load all data from the rec file once in the main process
    full_data, fields = rec.get_data(file=recfile, selected_fields=[key1, key2, key_mol2_1, key_mol2_2])
    num_records = len(full_data[fields[0]]) if fields else 0

    # Prepare arguments for multiprocessing pool
    task_args = []
    for i in range(num_records):
        smi1 = full_data[key1][i].replace("'", "") if key1 else None
        smi2 = full_data[key2][i].replace("'", "") if key2 else None
        mol2_1_path = full_data[key_mol2_1][i].replace("'", "") if key_mol2_1 else None
        mol2_2_path = full_data[key_mol2_2][i].replace("'", "") if key_mol2_2 else None
        task_args.append((smi1, smi2, mol2_1_path, mol2_2_path))
    
    similarities = [np.nan] * num_records # Initialize all with NaN
    async_results = []

    # Use multiprocessing Pool for robustness against segfaults
    with multiprocessing.Pool(processes=os.cpu_count()) as pool: # type: ignore
        for args in task_args:
            async_results.append(pool.apply_async(_compute_single_record_sim, (args,)))

        for i, res in enumerate(tqdm(async_results, desc="computing similarities (multiprocess)")):
            try:
                # Retrieve result for this specific task with a timeout of 30 seconds
                similarities[i] = res.get(timeout=30) 
            except multiprocessing.TimeoutError:
                # If a worker hangs due to timeout, mark as NaN
                # similarities[i] is already np.nan from initialization
                pass
            except BaseException: # Catch other BaseExceptions (e.g., segfaults leading to broken pipe)
                # similarities[i] is already np.nan from initialization
                pass

    outfilename = os.path.splitext(recfile)[0] + "_gesim" + ".rec.gz"
    
    # Write the output file
    with gzip.open(outfilename, "wt") as outgz:
        for i in tqdm(range(num_records), desc=f"writing file: {outfilename}"):
            # Reconstruct the current record as a dictionary
            current_record_dict = {field: full_data[field][i] for field in fields}
            # Add the computed similarity
            current_record_dict['sim'] = similarities[i]
            
            # Format and write the record using rec.dict_to_rec
            outgz.write(rec.dict_to_rec(current_record_dict))


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
