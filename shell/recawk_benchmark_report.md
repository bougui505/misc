# recawk: Database Querying Benchmark Report

This report benchmarks the performance and file size implications of using the new DuckDB/Parquet database backend in `recawk` compared to traditional gzip text files.

## Benchmark Environment
* **Dataset**: `/home/bougui/source/pandora/datasets/pairwise_tmscore.rec.gz`
* **Total Records**: 371,953 records
* **Total Keys**: 6 (`i`, `j`, `pdb1`, `pdb2`, `tmscore`, `distance`)

---

## 1. Storage Footprint Comparison

The Parquet format uses advanced Zstd compression and a columnar layout, allowing it to compress record data even better than standard gzip text files.

| Format | File Size | Size Ratio | Compression |
| :--- | :--- | :--- | :--- |
| **`pairwise_tmscore.rec.gz`** (Original) | **5.3 MB** | 1.00x | Gzip (Standard) |
| **`pairwise_tmscore.parquet`** (DuckDB/Parquet) | **3.4 MB** | **0.64x** | Zstd (Column-oriented) |

> [!TIP]
> Converting to `.parquet` actually **reduces your storage footprint by 36%** compared to `.rec.gz`!

---

## 2. Query Speed Performance

We evaluated the query time (in seconds) under a filtered subset scenario using the database-level `-w` / `--where` engine.

### Scenario: Filtered Subset Query
*Query: Filtering for records with high structural alignment (`tmscore > 0.8`) and printing the PDB IDs.*

| Query Type | Command | Execution Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Text Scan** | `zcat data.rec.gz \| recawk 'if (rec["tmscore"] > 0.8) {print ...}'` | **1.05s** | *Baseline* |
| **Parquet DB** | `recawk -w "CAST(tmscore AS FLOAT) > 0.8" '{print ...}' data.parquet` | **0.20s** | **5.2x faster** |

---

## 3. Key Takeaways

> [!IMPORTANT]
> The DuckDB/Parquet integration provides the best of both worlds:
> 1. **36% space savings** over compressed `.rec.gz` files.
> 2. **5x+ query speedups** on filtered lookups.
> 3. Zero temporary file generation (no disk decompression overhead).

---

## Usage Quick Start

### Create a Database
Convert any record file to a Parquet database:
```bash
recawk --todb data.rec.gz
# Creates data.parquet automatically
```

### Querying the Database
Seamlessly query the database file using identical AWK syntax:
```bash
recawk -w "CAST(tmscore AS FLOAT) > 0.8" '{print rec["pdb1"], rec["pdb2"]}' data.parquet
```
