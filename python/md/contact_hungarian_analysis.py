#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Nov  5 16:28:28 2025

import csv
import gzip
import multiprocessing
import os

import MDAnalysis
try:
    from MDAnalysis.guesser.tables import vdwradii
except ImportError:
    from MDAnalysis.topology.tables import vdwradii

# Update vdwradii for robustness (support case-insensitive lookups and missing elements like Fe)
for k, v in list(vdwradii.items()):
    vdwradii[k.lower()] = v
    vdwradii[k.capitalize()] = v
for element, radius in [('Fe', 1.8), ('Zn', 1.39), ('Cl', 1.75)]:
    for k in (element.upper(), element.lower(), element.capitalize()):
        vdwradii[k] = radius

import numpy as np
import typer
from MDAnalysis.core.groups import AtomGroup
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm


# Add a helper function for parallel processing
def _compute_chunk_distances(args):
    """
    Compute distances between two chunks for parallel processing.
    """
    chunk1, chunk2, i, j, labels, stride = args
    
    # Compute the distance matrix
    cdist_matrix = cdist(chunk1[::stride], chunk2[::stride])

    # Normalize the cdist_matrix:
    cdist_matrix /= np.sqrt(chunk1.shape[1])
    
    # Use linear sum assignment (Hungarian algorithm)
    row_ind_hungarian, col_ind_hungarian = linear_sum_assignment(cdist_matrix)

    # Make another assignment: Assign the minimal distance and not necessarily one to one as in the hungarian
    row_ind_min1 = np.arange(cdist_matrix.shape[0])
    col_ind_min1 = np.argmin(cdist_matrix, axis=1)
    row_ind_min2 = np.argmin(cdist_matrix, axis=0)
    col_ind_min2 = np.arange(cdist_matrix.shape[1])
    row_ind_min = np.concatenate((row_ind_min1, row_ind_min2))
    col_ind_min = np.concatenate((col_ind_min1, col_ind_min2))

    # Create hungarian directory if it doesn't exist
    os.makedirs("hungarian", exist_ok=True)
    
    # Write results to CSV file for Hungarian algorithm
    with gzip.open(f"hungarian/hungarian_distances_{labels[i]}_to_{labels[j]}.csv.gz", "wt", newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(["Frame_in_"+labels[i], "Frame_in_"+labels[j], "Distance"])
        for r, c in zip(row_ind_hungarian, col_ind_hungarian):
            writer.writerow([r, c, cdist_matrix[r, c]])
    
    # Write results to CSV file for minimal distance assignment
    with gzip.open(f"hungarian/minimal_distances_{labels[i]}_to_{labels[j]}.csv.gz", "wt", newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(["Frame_in_"+labels[i], "Frame_in_"+labels[j], "Distance"])
        for r, c in zip(row_ind_min, col_ind_min):
            writer.writerow([r, c, cdist_matrix[r, c]])
    
    return (i, j)

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.command()
def main(
    pdb: str = typer.Argument(..., help="Path to the PDB file."),
    dcd: str = typer.Argument(..., help="Path to the DCD trajectory file."),
    ligand_selection: str = typer.Argument(..., help="Ligand selection to define the pocket."),
    d_cut: float = typer.Option(6.0, help="Distance cutoff in angstrom for the pocket definition"),
    start_frame: int = typer.Option(0, help="Start frame of the trajectory segment used to define the pocket residues."),
    stop_frame: int = typer.Option(-1, help="Stop frame of the trajectory segment used to define the pocket residues (exclusive). -1 means the end of the trajectory."),
    remove_non_polar_h: bool = typer.Option(True, help="Remove non-polar hydrogens from the universe."),
    chunk_file: str = typer.Option(None, help="File containing the number of frames for each section of the trajectory. Useful to define different trajectories in a concatenated trajectory. The script will compute the Hungarian and minimal distance mappings pairwisely between each section. If not given, no pairwise calculation is performed. The chunk_file is a text file with each line describing one chunk with format: 'textual_description nbr of frames', e.g. 'no_trunc 50000'"),
    stride: int = typer.Option(1, help="Stride to apply to compute the Hungarian algorithm."),
    max_processes: int = typer.Option(None, help="Maximum number of processes to use for parallel computation. If not given, uses all available CPUs.")
):
    """
    Compute pairwise Hungarian and minimal distance alignments between sections
    of a trajectory based on pocket-ligand contact descriptors.
    """
    # Initialize variables for descriptors
    descriptors = None

    # 2. Compute raw descriptors if not loaded
    if descriptors is None:
        print("Computing raw descriptors.")
        u = MDAnalysis.Universe(pdb, dcd, to_guess=["types", "masses", "bonds"])
        print("##########################################")
        print(f"Number of frames: {u.trajectory.n_frames}")
        print(f"Total time: {u.trajectory.totaltime:.2f} ps")
        print(f"Number of atoms: {len(u.atoms)}")

        excluded_atoms = None
        if remove_non_polar_h:
            # Select non-polar hydrogens from the entire universe
            non_polar_h_selection = "name H* and bonded (name C*)"
            non_polar_hydrogens = u.select_atoms(non_polar_h_selection)

            if len(non_polar_hydrogens) > 0:
                print(f"Removing {len(non_polar_hydrogens)} non-polar hydrogens from the universe...")
                excluded_atoms = non_polar_hydrogens
                print(f"Effective number of atoms for analysis after excluding non-polar H: {len(u.atoms) - len(excluded_atoms)}")
            else:
                print("No non-polar hydrogens found in the universe to remove.")

        pocket = get_pocket(u, start_frame, stop_frame, ligand_selection, d_cut, pdb, dcd, excluded_atoms=excluded_atoms)
        ligand = u.select_atoms(ligand_selection)
        if excluded_atoms is not None:
            ligand = ligand - excluded_atoms
        descriptors = get_descriptors(u, 0, -1, pocket, ligand, d_cut=d_cut)
        print("Raw descriptors computed.")

    # 3. Process chunks if chunk_file is provided
    if chunk_file is not None:
        print("Processing trajectory chunks.")
        
        with open(chunk_file, "r") as cf:
            chunks = []
            labels = []
            c_prev = 0
            for line in cf:
                l, c = line.strip().split()
                c = c_prev + int(c)
                labels.append(l)
                chunks.append(c)
                c_prev = c

        # Split descriptors into chunks
        desc_chunks = np.split(descriptors, chunks[:-1])
        
        # Prepare arguments for parallel processing
        args_list = []
        n_chunks = len(desc_chunks)
        for i in range(n_chunks):
            for j in range(i+1, n_chunks):
                chunk1 = desc_chunks[i]
                chunk2 = desc_chunks[j]
                args_list.append((chunk1, chunk2, i, j, labels, stride))
        
        # Process chunks in parallel
        if max_processes is not None:
            nprocs = min(max_processes, len(args_list))
        else:
            nprocs = min(multiprocessing.cpu_count(), len(args_list))
        print(f"Computing pairwise distances in parallel with {nprocs} processes.")
        if nprocs > 1:
            with multiprocessing.Pool(processes=nprocs) as pool:
                list(tqdm(pool.imap_unordered(_compute_chunk_distances, args_list), 
                         total=len(args_list), desc="Computing pairwise distances"))
        else:
            # Fallback to sequential processing if only one process
            for i in tqdm(range(n_chunks), desc="Computing pairwise distances"):
                for j in range(i+1, n_chunks):
                    chunk1 = desc_chunks[i]
                    chunk2 = desc_chunks[j]
                    _compute_chunk_distances((chunk1, chunk2, i, j, labels, stride))


def _get_pocket_chunk(args):
    """
    Helper function for parallel processing of a trajectory chunk.
    Loads its own universe, processes frames, and returns a set of pocket residues.
    """
    pdb_path, dcd_path, start_frame_chunk, stop_frame_chunk, ligand_selection, d_cut, excluded_atoms_ids = args

    u_worker = MDAnalysis.Universe(pdb_path, dcd_path, to_guess=["types", "masses", "bonds"])
    ligand_worker = u_worker.select_atoms(ligand_selection)

    excluded_atoms_worker = None
    if excluded_atoms_ids is not None:
        excluded_atoms_worker = u_worker.select_atoms(f"id {' '.join(map(str, excluded_atoms_ids))}")
        ligand_worker = ligand_worker - excluded_atoms_worker

    if not ligand_worker:
        # Return an empty set if ligand selection is empty in this worker's universe
        return set()

    pocket_residues_worker = set()
    # Pre-compute ligand_atom_ids_str as it's constant for the chunk
    ligand_atom_ids_str = " ".join(map(str, ligand_worker.ids))

    for ts in u_worker.trajectory[start_frame_chunk:stop_frame_chunk]:
        contacts = u_worker.select_atoms(f"protein and around {d_cut} id {ligand_atom_ids_str}")
        if excluded_atoms_worker is not None:
            contacts = contacts - excluded_atoms_worker
        for res in contacts.residues:
            # Store a picklable representation of the residue (resname, resid, segid)
            pocket_residues_worker.add((res.resname, res.resid, res.segid))
    return pocket_residues_worker


def get_pocket(universe, start_frame, stop_frame, ligand_selection, d_cut, pdb_path, dcd_path, excluded_atoms: AtomGroup = None):
    """
    Get the protein residues that are around the ligand with d_cut cutoff (in angstrom)
    in the trajectory from start_frame to stop_frame.
    If an atom of a residue appears at least once during the trajectory keep this residue
    in the pocket definition.
    """
    total_frames = universe.trajectory.n_frames
    if stop_frame == -1:
        stop_frame = total_frames

    # Prepare excluded_atoms_ids for passing to workers (or for sequential path's internal use)
    excluded_atoms_ids_to_pass = None
    if excluded_atoms is not None:
        excluded_atoms_ids_to_pass = excluded_atoms.ids.tolist()

    pocket_residues_set = set() # Use a distinct name for the set of residues

    nprocs = os.cpu_count()
    if nprocs > 1:
        print(f"Parallelizing pocket definition with {nprocs} processes.")
        frame_indices = np.arange(start_frame, stop_frame)
        chunks = np.array_split(frame_indices, nprocs)

        worker_args = []
        for chunk in chunks:
            if len(chunk) > 0: # Only create arguments for non-empty chunks
                worker_args.append((pdb_path, dcd_path, chunk[0], chunk[-1] + 1,
                                    ligand_selection, d_cut, excluded_atoms_ids_to_pass))

        with multiprocessing.Pool(processes=nprocs) as pool:
            # Use imap_unordered for potentially faster processing of results
            for worker_result in tqdm(pool.imap_unordered(_get_pocket_chunk, worker_args),
                                      total=len(worker_args), desc="Defining pocket (parallel chunks)"):
                pocket_residues_set.update(worker_result)
    else: # nprocs == 1 (sequential processing)
        ligand_for_contacts = universe.select_atoms(ligand_selection)
        if excluded_atoms is not None:
            ligand_for_contacts = ligand_for_contacts - excluded_atoms

        if not ligand_for_contacts:
            print(f"Warning: No atoms found for ligand selection: {ligand_selection}. This may be due to hydrogens being removed from ligand.")
            return universe.select_atoms("") # Return empty AtomGroup

        ligand_atom_ids_str = " ".join(map(str, ligand_for_contacts.ids))

        for ts in tqdm(universe.trajectory[start_frame:stop_frame], desc="Defining pocket (sequential)"):
            contacts = universe.select_atoms(f"protein and around {d_cut} id {ligand_atom_ids_str}")
            if excluded_atoms is not None:
                contacts = contacts - excluded_atoms

            for res in contacts.residues:
                pocket_residues_set.add((res.resname, res.resid, res.segid))

    print(f"Number of unique pocket residues identified: {len(pocket_residues_set)}")

    # Make an AtomGroup of the pocket residues.
    # Initialize an empty AtomGroup belonging to the universe.
    pocket_atomgroup = universe.select_atoms("")
    # Iterate through the collected residue identifiers and add their atoms from the main universe
    for resname, resid, segid in pocket_residues_set:
        # Select atoms from the main universe using the identifiers
        res_atoms_main_universe = universe.select_atoms(f"resname {resname} and resid {resid} and segid {segid}")
        pocket_atomgroup = pocket_atomgroup.union(res_atoms_main_universe)
    print(f"Number of atoms in pocket_atomgroup (after combining residue AtomGroups): {len(pocket_atomgroup)}")

    if excluded_atoms is not None:
        # Remove excluded atoms (e.g., non-polar hydrogens) from the pocket AtomGroup
        pocket_atomgroup = pocket_atomgroup - excluded_atoms
        print(f"Number of atoms in pocket_atomgroup (after removing excluded atoms): {len(pocket_atomgroup)}")

    # Renaming for clarity and to avoid confusion with the set of residues
    pocket_residues_atomgroup = pocket_atomgroup

    return pocket_residues_atomgroup


def get_descriptors(universe, start_frame, stop_frame, pocket_residues, ligand, d_cut):
    # Calculate the expected length of a single descriptor vector
    # This will be 0 if either pocket_residues or ligand has no atoms
    descriptor_length = len(pocket_residues.positions) * len(ligand.positions)

    # Determine the number of frames to process
    total_frames = universe.trajectory.n_frames
    if stop_frame == -1:
        effective_stop_frame = total_frames
    else:
        effective_stop_frame = min(stop_frame, total_frames)

    num_frames_to_process = max(0, effective_stop_frame - start_frame)

    # Handle cases where no frames are processed or descriptors have zero length
    if num_frames_to_process == 0:
        # If no frames to process, return an empty array with 0 rows and `descriptor_length` columns
        return np.empty((0, descriptor_length), dtype=np.float32)

    if descriptor_length == 0:
        # If descriptor length is 0 (due to empty pocket/ligand positions),
        # return an array with `num_frames_to_process` rows and 0 columns
        print("Warning: Pocket or ligand positions are empty. Descriptors will have zero length.")
        return np.empty((num_frames_to_process, 0), dtype=np.float32)

    # If we reach here, we expect num_frames_to_process > 0 and descriptor_length > 0
    # Pre-allocate the array for efficiency to avoid Python list overhead and repeated array reallocations.
    descriptors_array = np.empty((num_frames_to_process, descriptor_length), dtype=np.float32)

    for i, ts in enumerate(tqdm(universe.trajectory[start_frame:effective_stop_frame], desc="Computing descriptors")):
        descriptors_array[i, :] = cdist(pocket_residues.positions, ligand.positions).flatten().astype(np.float32)

    print(f"Shape of the descriptor array before filtering columns: {descriptors_array.shape}")
    # Filter out columns with systematic distances higher than d_cut
    descriptors_array = descriptors_array[:,(descriptors_array < d_cut).any(axis=0)]
    print(f"Shape of the descriptor array after filtering columns: {descriptors_array.shape}")
    return descriptors_array


if __name__ == "__main__":
    app()
