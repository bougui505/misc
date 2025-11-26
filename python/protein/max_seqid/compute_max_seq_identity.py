#!/usr/bin/env python3
import os
from functools import partial
from multiprocessing import Pool
from typing_extensions import Annotated

import typer
from Bio.Align import PairwiseAligner
from tqdm import tqdm


def read_fasta(fasta_file: str) -> dict[str, str]:
    """
    Reads sequences from a FASTA file into a dictionary.
    Keys are sequence IDs (up to the first space), values are the sequences.
    """
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    return sequences

def calculate_sequence_identity(seq1, seq2):
    """
    Calculates sequence identity between two sequences after performing a global alignment.
    Identity is defined as the number of matching characters divided by the length of the alignment.
    """
    if not seq1 or not seq2:
        return 0.0

    # Initialize the aligner with parameters matching pairwise2.align.globalxx defaults.
    aligner = PairwiseAligner()
    aligner.mode = 'global' # Needleman-Wunsch global alignment
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    # Perform global alignment.
    alignments = aligner.align(seq1, seq2)

    if not alignments:
        return 0.0

    # Take the first (best) alignment from the iterator.
    first_alignment = next(alignments)

    # Access the aligned sequences as strings.
    aligned_seq1 = str(first_alignment.target)
    aligned_seq2 = str(first_alignment.query)

    # Calculate identities by comparing characters.
    matches = 0
    gap_char = '-' # Standard gap character in Biopython alignments
    for char1, char2 in zip(aligned_seq1, aligned_seq2):
        if char1 == char2 and char1 != gap_char:
            matches += 1

    # The aligned length is the length of either aligned sequence.
    aligned_length = len(aligned_seq1)

    if aligned_length == 0:
        return 0.0

    return matches / aligned_length

def worker(test_seq_item, reference_sequences):
    """
    Worker function to compute the maximum sequence identity for a single test sequence
    against all reference sequences.
    """
    test_id, test_seq = test_seq_item
    max_identity = 0.0

    for ref_id, ref_seq in reference_sequences.items():
        identity = calculate_sequence_identity(test_seq, ref_seq)
        if identity > max_identity:
            max_identity = identity
    return test_id, max_identity

app = typer.Typer(
    help="Compute maximum sequence identity for test sequences relative to reference sequences in parallel."
)


@app.command()
def main(
    test_fasta: Annotated[
        str, typer.Argument(help="Path to the FASTA file containing test sequences.")
    ],
    ref_fasta: Annotated[
        str, typer.Argument(help="Path to the FASTA file containing reference sequences.")
    ],
    jobs: Annotated[
        int,
        typer.Option(
            "-j",
            "--jobs",
            help=f"Number of parallel jobs (default: {os.cpu_count()}).",
        ),
    ] = os.cpu_count(),
):
    print(f"Reading test sequences from {test_fasta}...")
    test_sequences = read_fasta(test_fasta)
    print(f"Found {len(test_sequences)} test sequences.")

    print(f"Reading reference sequences from {ref_fasta}...")
    reference_sequences = read_fasta(ref_fasta)
    print(f"Found {len(reference_sequences)} reference sequences.")

    if not test_sequences:
        print("Error: Test FASTA file is empty or could not be read. Exiting.")
        raise typer.Exit(code=1)
    if not reference_sequences:
        print("Error: Reference FASTA file is empty or could not be read. Exiting.")
        raise typer.Exit(code=1)

    print(f"Computing maximum sequence identity with {jobs} parallel jobs...")
    test_seq_items = list(test_sequences.items())

    # Use functools.partial to pass the static reference_sequences to each worker
    worker_partial = partial(worker, reference_sequences=reference_sequences)

    print(f"Processing {len(test_seq_items)} test sequences...")
    with Pool(processes=jobs) as pool:
        results = list(tqdm(pool.imap(worker_partial, test_seq_items), total=len(test_seq_items)))

    print("\nMaximum Sequence Identities:")
    for test_id, max_identity in results:
        print(f"{test_id}: {max_identity:.4f}")


if __name__ == "__main__":
    app()
