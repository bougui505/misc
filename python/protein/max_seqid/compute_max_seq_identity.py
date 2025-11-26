#!/usr/bin/env python3
import argparse
import os
from functools import partial
from multiprocessing import Pool


def read_fasta(fasta_file):
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
    Calculates sequence identity between two sequences using a simple character-by-character
    comparison over the length of the shorter sequence. This approach does not perform
    a full sequence alignment (e.g., Needleman-Wunsch or Smith-Waterman) and thus
    might not capture the 'maximum' identity in cases with indels or complex rearrangements.
    """
    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0

    matches = 0
    for i in range(min_len):
        if seq1[i] == seq2[i]:
            matches += 1

    return matches / min_len

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

def main():
    parser = argparse.ArgumentParser(
        description="Compute maximum sequence identity for test sequences relative to reference sequences in parallel."
    )
    parser.add_argument("test_fasta", help="Path to the FASTA file containing test sequences.")
    parser.add_argument("ref_fasta", help="Path to the FASTA file containing reference sequences.")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help=f"Number of parallel jobs (default: {os.cpu_count()}).")
    args = parser.parse_args()

    print(f"Reading test sequences from {args.test_fasta}...")
    test_sequences = read_fasta(args.test_fasta)
    print(f"Found {len(test_sequences)} test sequences.")

    print(f"Reading reference sequences from {args.ref_fasta}...")
    reference_sequences = read_fasta(args.ref_fasta)
    print(f"Found {len(reference_sequences)} reference sequences.")

    if not test_sequences:
        print("Error: Test FASTA file is empty or could not be read. Exiting.")
        return
    if not reference_sequences:
        print("Error: Reference FASTA file is empty or could not be read. Exiting.")
        return

    print(f"Computing maximum sequence identity with {args.jobs} parallel jobs...")
    test_seq_items = list(test_sequences.items())

    # Use functools.partial to pass the static reference_sequences to each worker
    worker_partial = partial(worker, reference_sequences=reference_sequences)

    with Pool(processes=args.jobs) as pool:
        results = pool.map(worker_partial, test_seq_items)

    print("\nMaximum Sequence Identities:")
    for test_id, max_identity in results:
        print(f"{test_id}: {max_identity:.4f}")

if __name__ == "__main__":
    main()
