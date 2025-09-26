# Repository Overview

This README provides a quick overview of the files within this repository.

## python/C/logit/test.py
This file appears to be a test script or a script performing numerical operations, possibly related to logit functions, given the use of `numpy.linspace`.

## python/Grid3/adjmat.py
Contains functions for handling adjacency matrices, grid movement, and finding the shortest path, likely within a 3D grid context.

## python/Grid3/gradient.py
Defines the `Gradient` class, suggesting functionality for gradient-related calculations or representations on a grid.

## python/Grid3/mrc.py
Includes a `save_density` function, indicating it handles saving density data, possibly to a file format like MRC.

## python/HDF5/hdf5set.py
Provides utilities for HDF5 dataset manipulation, including generating random keys for data storage and retrieval.

## python/TMalign/TMalign.cpp
A core C++ file implementing the TM-align algorithm for protein structural alignment. It includes functions for array management, parsing PDB files, performing rotations, searching for optimal alignments, and calculating TM-scores.

## python/TMalign/USalign.cpp
Another significant C++ file, likely part of the US-align suite. It features templated classes for inter-process communication (pstreams), error handling, array management, and various alignment/docking algorithms such as `MMalign`, `MMdock`, and `mTMalign`.

## python/Timer.py
Defines a `Timer` class for measuring execution time, providing `stop` and `reset` methods for convenient timing of code blocks.

## python/annoy/NNindex.py
Contains `NNindex` and `Mapping` classes, providing functionality for approximate nearest neighbor indexing using Annoy, along with mapping between data indices and names.

## python/debug.py
Defines a `Debugger` class, indicating it serves as a utility for debugging purposes. It includes a main block for testing.

## python/eta.py
Implements an `ETA` (Estimated Time of Arrival) class, which can be used to track progress and estimate remaining time for iterative processes.

## python/gym/loader.py
A utility `Loader` class for saving and loading OpenAI Gym environments, particularly useful for integrating with libraries like Stable Baselines.

## python/gym/mullerenv.py
Defines an OpenAI Gym environment based on the Muller potential, including functions for potential calculation, reward mapping, and environment interaction (`step`, `reset`).

## python/gym/predict_muller.py
A script likely used for predicting or visualizing outcomes related to the Muller potential environment, with flags for saving and showing results.

## python/holoviews/holoviews_app.py
Sets up and runs a HoloViews/Panel application server, suggesting interactive data visualization capabilities.

## python/interpolation.py
Provides utility functions `format_line` and `format_output` for formatting text lines and output, potentially used in data processing or displaying interpolation results.

## python/list_utils.py
Contains a `flatten` function, a common utility for converting a list of lists into a single flat list.

## python/modeller/build.py
Offers tools for building and optimizing protein models using the MODELLER library, including a `System` class for sequential residue addition and minimization.

## python/mols/graph/mol_to_graph.py
Central to converting molecular structures (SMILES) into graph representations suitable for machine learning. It includes functions for atom featurization, graph conversion, and a `MolDataset` class for handling molecular datasets.

## python/mols/graph/utils.py
Contains `get_mapping`, likely providing utility functions related to molecular graphs, possibly for mapping or indexing.

## python/mols/ligrmsd.py
Includes a `get_coords` function, presumably for extracting atomic coordinates from molecular objects, often a precursor to calculating ligand RMSD.

## python/mols/rdkit_fix.py
Another file with a `get_coords` function, likely for extracting coordinates from molecular objects, possibly addressing RDKit-specific behaviors or providing utility functions.

## python/muller_potential.py
Defines functions `muller_potential` and `muller_mat` for calculating and representing the Muller potential energy surface.

## python/netCDF4/netcdf4set.py
Provides an `N4set` class for managing NetCDF4 datasets, including methods for adding single data items or batches, and retrieving random keys.

## python/npcli/np.py
A command-line interface utility, likely for NumPy-related operations, specifically for formatting output lines.

## python/openmm/md.py
Contains functions `add_plumed_forces` and `run`, suggesting its role in setting up and executing molecular dynamics simulations using OpenMM, with potential integration with PLUMED.

## python/openmm/topology.py
Defines the `MetaSystem` class, which extends OpenMM's topology functionalities to build and manipulate molecular systems, particularly for proteins by adding residues like GLY and OXT.

## python/protein/VAE/utils.py
Contains utility functions `get_dmat`, `compute_pad`, and `pad`, likely used in the context of Variational Autoencoders (VAEs) for protein data, potentially involving distance matrix calculations.

## python/protein/coords_loader.py
Provides a `get_coords` function for loading protein coordinates from PDB files, offering various selection, splitting, and transformation options.

## python/protein/count_beta_strands.py
Focuses on analyzing protein secondary structure, specifically counting beta strands and formatting the output.

## python/protein/depict/depict.py
Includes a `get_mapping` function, likely used for mapping elements in the context of protein depiction.

## python/protein/get_chains.py
Provides a `get_chains` function to extract chain identifiers from a given PDB file.

## python/protein/interpred/utils.py
Contains a `get_coords` utility, likely for extracting protein coordinates in the context of interaction prediction.

## python/protein/spherical.py
Defines the `Internal` class to calculate and store internal spherical coordinates of a protein C-alpha trace, with a method to write the data to a file.

## python/protein/sscl/BLASTloader.py
Implements a `PDBdataset` class as a PyTorch Dataset for PDB structures, probably used for protein secondary structure classification (SSCL) or self-supervised learning, potentially leveraging BLAST data.

## python/protein/sscl/utils.py
Contains utility functions `get_coords`, `compute_pad`, and `pad` for protein secondary structure classification (SSCL), including coordinate extraction and padding operations.

## python/protein/sscl_geometric/BLASTloader.py
Similar to `python/protein/sscl/BLASTloader.py`, this file also defines a `PDBdataset` as a PyTorch Dataset but is tailored for graph-based protein representations using `torch_geometric.data.Data` objects.

## python/pyawk.py
Provides a `format_line` function, suggesting it's a Python utility for line processing and formatting, potentially inspired by the AWK command-line tool.

## python/pytorch/Dataset_gz.py
Implements a `Dataset` class as a PyTorch Dataset for handling gzipped text files, enabling efficient data loading for machine learning models.

## python/pytorch/Dataset_txt.py
Defines a `Dataset` class as a PyTorch Dataset for reading and processing plain text files line by line, suitable for various data loading tasks.

## python/pytorch/pytorch_benchmark_script.py
A script containing `batched_dot_mul_sum` and `batched_dot_bmm` functions, designed for benchmarking different implementations of batched dot product operations in PyTorch.

## python/rec.py
A comprehensive utility for record-based data manipulation, including functions for formatting, reading from columns files, data selection, adding properties, and merging dictionaries.

## python/recutils.py
Provides utilities for loading and manipulating data in a record-like format, including `get_item`, `add_item`, and `load` functions.

## python/shelve/shelveset.py
Defines a `Shelveset` class for managing Python `shelve` databases, allowing the storage and retrieval of arbitrary Python objects with random keys.

## python/sliding.py
Implements the `Sliding_op` class for applying a function over a sliding window on 1D input data, with options for padding.

## python/timeseries.py
Contains a `split` function, a utility for dividing time series data into specified intervals based on a delta time.

## python/toymodels/montecarlo.py
Provides functions for implementing Monte Carlo simulations, including defining potential energy functions, performing moves, and plotting potential energy landscapes and distributions.

## shell/tmscore/np.py
A utility script, likely used within a shell environment related to TMscore, containing a `format_line` function for formatting NumPy array outputs.
