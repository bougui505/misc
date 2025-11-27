# Compute Maximum Sequence Identity

This script computes the maximum sequence identity for a set of test sequences against a set of reference sequences using parallel processing. It's useful for tasks like identifying the closest known homolog for a protein sequence.

## Features

- Reads sequences from FASTA files.
- Calculates sequence identity using global pairwise alignment (Needleman-Wunsch).
- Supports parallel execution for faster processing of large datasets.
- Command-line interface using `typer`.

## Installation

This script requires `biopython` and `typer`. You can install them using pip:

```bash
pip install biopython typer
```

## Usage

To run the script, you need to provide paths to two FASTA files: one for the test sequences and one for the reference sequences.

```bash
python python/protein/max_seqid/compute_max_seq_identity.py <TEST_FASTA_FILE> <REFERENCE_FASTA_FILE> [OPTIONS]
```

### Arguments

- `TEST_FASTA_FILE`: Path to the FASTA file containing test sequences.
- `REFERENCE_FASTA_FILE`: Path to the FASTA file containing reference sequences.

### Options

- `-j`, `--jobs INTEGER`: Number of parallel jobs to use for computation. By default, it uses all available CPU cores.

### Example

Let's assume you have `test.fasta` and `ref.fasta` in your current directory.

```bash
# Example test.fasta
# >test_seq1
# MASKLVLFGAAFLGAATLATVL
# >test_seq2
# MLVGSAAAFLGAATLAVVL

# Example ref.fasta
# >ref_protein_A
# MAKLVLFGAAFAGAATLATVL
# >ref_protein_B
# MLVGSAAGFLGAATLAAVL

python python/protein/max_seqid/compute_max_seq_identity.py test.fasta ref.fasta -j 4
```

This command will read sequences from `test.fasta` and `ref.fasta`, then compute the maximum sequence identity for each test sequence against all reference sequences using 4 parallel jobs. The results will be printed to the console.

## How it Works

1. **`read_fasta(fasta_file: str) -> dict[str, str]`**: Reads sequences from a FASTA file. Sequence IDs are parsed up to the first space.
2. **`calculate_sequence_identity(seq1, seq2) -> float`**: Computes the sequence identity between two input sequences. It performs a global alignment using `Bio.pairwise2.align.globalxx` and calculates identity as the number of matching characters (excluding gaps) divided by the length of the shorter original sequence.
3. **`worker(test_seq_item, reference_sequences) -> tuple[str, float]`**: A multiprocessing worker function that takes a single test sequence and compares it against all reference sequences to find the maximum identity.
4. **`main()`**: The main function orchestrates the reading of FASTA files, sets up a multiprocessing pool, and distributes the work among the `worker` functions. It then prints the results.

## Defining Sequence Identity

The sequence identity is calculated as:
$$
\text{Identity} = \frac{\text{Number of matching characters}}{\text{min(Length of Seq1, Length of Seq2)}}
$$
This definition ensures that the identity is relative to the length of the shorter sequence, providing a common metric in protein sequence comparison. Gaps introduced by alignment are not counted as matches.
