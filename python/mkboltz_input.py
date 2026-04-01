#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-04-01

import typer
import yaml
from pathlib import Path
from typing import List

app = typer.Typer(help="Generate a Boltz-2 YAML input file from biological sequences.")

def parse_fasta(fasta_path: Path):
    """Simple FASTA parser to extract sequences."""
    sequences = []
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences

def reverse_complement(seq: str) -> str:
    """Generates the complementary DNA strand."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    # Reverse the sequence and map to complement
    return "".join(complement.get(base, base) for base in reversed(seq.upper()))

@app.command()
def main(
    output: Path = typer.Option("input.yaml", "--out", "-o", help="Path to save the generated YAML."),
    protein: List[Path] = typer.Option([], "--protein", "-p", help="Path to protein FASTA files. Multiple files can be given by repeating the -p option."),
    dna: List[Path] = typer.Option([], "--dna", "-d", help="Path to DNA FASTA files. Multiple files can be given by repeating the -d option. The reverse complement sequence will be automatically built."),
    rna: List[Path] = typer.Option([], "--rna", "-r", help="Path to RNA FASTA files. Multiple files can be given by repeating the -r option."),
    smiles: List[str] = typer.Option([], "--smiles", "-s", help="SMILES strings for ligands. Multiple files can be given by repeating the -s option."),
):
    """
    Generate a Boltz-2 compatible YAML file.
    """
    yaml_data = {
        "sequences": []
    }
    
    # Simple ID generator (A, B, C...)
    id_counter = 0
    def get_next_id():
        nonlocal id_counter
        label = chr(65 + (id_counter % 26)) * (1 + id_counter // 26)
        id_counter += 1
        return label

    # Process Polymers (Protein, DNA, RNA)
    for category, paths in [("protein", protein), ("dna", dna), ("rna", rna)]:
        for path in paths:
            seqs = parse_fasta(path)
            for s in seqs:
                entry = {category: {"id": get_next_id(), "sequence": s}}
                # Boltz defaults proteins to 'msa' server if not specified
                yaml_data["sequences"].append(entry)
                if category == "dna":
                    # Generate complement sequence
                    cs = reverse_complement(s)
                    entry = {category: {"id": get_next_id(), "sequence": cs}}
                    yaml_data["sequences"].append(entry)

    # Process Ligands
    for s in smiles:
        yaml_data["sequences"].append({
            "ligand": {
                "id": get_next_id(),
                "smiles": s
            }
        })

    if not yaml_data["sequences"]:
        typer.echo("Error: No sequences provided.", err=True)
        raise typer.Exit(1)

    with open(output, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    
    typer.echo(f"Successfully wrote Boltz-1 config to {output}")

if __name__ == "__main__":
    app()
