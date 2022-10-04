## Install

From the deb package on Github singularity repo

```
sudo apt install ./singularity-ce_3.10.2-focal_amd64.deb
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
