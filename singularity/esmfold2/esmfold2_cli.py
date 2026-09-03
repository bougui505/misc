#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from transformers import AutoModel
from esm.models.esmfold2 import (
    ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
)

def main():
    parser = argparse.ArgumentParser(description="CLI for running Biohub's ESMFold2 locally.")
    parser.add_argument("-s", "--sequence", type=str, help="Amino acid sequence to fold")
    parser.add_argument("-o", "--output", type=str, default="output.cif", help="Path to save the output MMCIF file")
    args = parser.parse_args()

    if not args.sequence:
        print("Error: Please provide a sequence using --sequence or -s")
        return

    print("Loading ESMFold2...")
    model = AutoModel.from_pretrained("biohub/ESMFold2", trust_remote_code=True).cuda().eval()
    
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="target", sequence=args.sequence)]
    )

    print(f"Folding sequence of length {len(args.sequence)}...")
    with torch.inference_mode():
        result = ESMFold2InputBuilder().fold(
            model, 
            spi, 
            num_loops=10, 
            num_sampling_steps=200, 
            num_diffusion_samples=1, 
            seed=0
        )

    print(f"Saving prediction to {args.output}")
    with open(args.output, "w") as f:
        f.write(result.complex.to_mmcif())
    print("Done!")

if __name__ == "__main__":
    main()
