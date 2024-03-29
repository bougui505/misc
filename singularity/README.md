## Install

From the deb package on Github singularity repo

```
wget https://github.com/apptainer/apptainer/releases/download/v1.1.2/apptainer_1.1.2_amd64.deb \
    && sudo apt-get install ./apptainer_1.1.2_amd64.deb
```

## `builder.sh` helper script for building `SIF` images

```
❯ ./builder.sh -h           
Help message
    -h, --help print this help message and exit
    -d, --def def file to build
    -f, --force The --force option will delete and overwrite an existing Singularity image without presenting the normal interactive prompt
```

## `run.sh` helper script to run a command in a singularity container

```
❯ run -h                  

run the COMMAND in a singularity container
run [-i image] -- COMMAND

Help message
    -h, --help print this help message and exit
    -i, --image singularity sif image to use (default is pytorch.sif)
    -- COMMAND
```

## Build images from scratch
See: https://docs.sylabs.io/guides/3.5/user-guide/build_a_container.html#building-containers-from-singularity-definition-files

```
sudo singularity build debian.sif debian.def
```

## Interactive shell:

```
singularity shell debian.sif
```

## To run a command:
```
singularity run pytorch.sif python3 -c 'import torch; print(torch.cuda.is_available())'
```

To load nvidia drivers:
```
singularity run --nv pytorch.sif python3 -c 'import torch; print(torch.cuda.is_available())'
```

To bind libgl (from DCV) for pymol with `nv`
```
singularity run --nv -B /usr/lib64/libGL.so.1.7.0:/var/lib/dcv-gl/lib64/libGL_SYS.so.1.0.0 pytorch.sif python3 -c 'from pymol import cmd'
```

To run in the CWD
```
singularity run --pwd $(pwd) /ld18-1006/work/bougui/pytorch.sif pwd
```

To bind a local directory (here: `/ld18-1006`):
```
singularity run -B /ld18-1006 --pwd $(pwd) /ld18-1006/work/bougui/pytorch.sif pwd
```

## Remote builder

```
singularity remote login
```

Tokens are generated from:
https://cloud.sylabs.io/auth/tokens

