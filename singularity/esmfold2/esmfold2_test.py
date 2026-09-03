#!/usr/bin/env python3
import torch
from transformers import AutoModel
from esm.models.esmfold2 import (
    ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
)

# Load the model from the cached HF directory
print("Loading ESMFold2 model...")
model = AutoModel.from_pretrained("biohub/ESMFold2", trust_remote_code=True).cuda().eval()

# Define a test sequence
test_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
print(f"Folding sequence (length {len(test_sequence)})...")

spi = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence=test_sequence)
    ]
)

# Fold the protein
with torch.inference_mode():
    result = ESMFold2InputBuilder().fold(
        model, 
        spi, 
        num_loops=10, 
        num_sampling_steps=200, 
        num_diffusion_samples=1, 
        seed=0
    )

# Save output
output_file = "result.cif"
print(f"Saving predicted structure to {output_file}...")
with open(output_file, "w") as f:
    f.write(result.complex.to_mmcif())

print("Prediction completed successfully!")
