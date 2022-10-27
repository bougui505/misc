#!/usr/bin/env zsh
sudo singularity build --force --sandbox extra.{sif,def} && sudo singularity build --force --sandbox shell.{sif,def} && sudo rsync -a -zz --update --info=progress2 -h --delete -e 'ssh -F /home/bougui/.ssh/config' shell.sif desk:~/source/misc/singularity/conda
