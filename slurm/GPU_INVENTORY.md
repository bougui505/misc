# Maestro GPU Inventory & Performance Specifications

This document outlines the GPU models available on the Maestro cluster, ranked by their overall performance/speed (from fastest to slowest).

## GPU Speed Rankings

The inventory in `gpu_stats.py` is sorted by GPU speed based on the following hierarchy:

| Rank | GPU Model | Architecture | Memory (VRAM) | FP32 Peak Performance | Tensor Performance / Core Specs | Key Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **l40s** | Ada Lovelace | 48 GB GDDR6 | ~91.6 TFLOPS | ~733 TFLOPS FP16, ~1,466 TFLOPS FP8 | AI Inference, fine-tuning, mixed AI & rendering |
| **2** | **A100** | Ampere | 40/80 GB HBM2e | ~19.5 TFLOPS | ~312 TFLOPS TF32 Tensor, high memory bandwidth | Large-scale AI training, HPC, MIG partitioning |
| **3** | **A40** | Ampere | 48 GB GDDR6 | ~37.4 TFLOPS | ~149.7 TFLOPS FP16 | Virtual workstations, rendering, media encoding |
| **4** | **rtx6000** | Turing | 24 GB GDDR6 | ~16.3 TFLOPS | ~130.5 TFLOPS Tensor | Workstation rendering, local/smaller AI development |
| **5** | **2g.48gb+gfx** | Ampere / Hopper | 48 GB (Slice) | Variable (Slice) | Multi-Instance GPU (MIG) slice (2/7ths compute) | Light/partitioned GPU compute & graphics |

---

## Detailed Model Overview

### 1. NVIDIA L40S (`l40s`)
*   **Architecture:** Ada Lovelace (4nm)
*   **Performance:** Fastest raw FP16/FP8 tensor core computing in the cluster. Highly optimized for massive throughput in AI inference and fine-tuning workloads.
*   **Memory:** 48 GB GDDR6 with 864 GB/s bandwidth.

### 2. NVIDIA A100 (`A100`)
*   **Architecture:** Ampere (7nm)
*   **Performance:** Standard for distributed deep learning. While its raw FP32 single-GPU TFLOPS are lower than newer architectures like Ada, its high-bandwidth memory (HBM2e) and robust NVLink scaling make it highly performant for memory-bound and multi-node AI training workloads.
*   **Memory:** 40 GB or 80 GB HBM2e (up to 2,039 GB/s bandwidth).

### 3. NVIDIA A40 (`A40`)
*   **Architecture:** Ampere (7nm)
*   **Performance:** Designed for server-side professional visualization. Offers solid FP32 graphics throughput and baseline AI training/inference capabilities.
*   **Memory:** 48 GB GDDR6 with 696 GB/s bandwidth.

### 4. NVIDIA RTX 6000 (`rtx6000`)
*   **Architecture:** Turing (12nm)
*   **Performance:** Older workstation-class GPU. Suited for general graphics rendering and legacy compute workloads.
*   **Memory:** 24 GB GDDR6.

### 5. Multi-Instance GPU Slice (`2g.48gb+gfx`)
*   **Description:** A virtual partition representing 2/7ths of a physical GPU's compute capability, allocated with 48 GB of memory and enabled graphics support (`+gfx`). Used for resource isolation and lightweight workloads.
