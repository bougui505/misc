#!/usr/bin/env zsh
until run --nv -B /ld18-1006 -- ./deminer.py --train --print_each 1 --n_epochs 5; do echo "Auto restart"; sleep 60; done
