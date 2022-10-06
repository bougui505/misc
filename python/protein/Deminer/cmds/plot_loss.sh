#!/usr/bin/env zsh
rsync desk:/ld18-1006/work/bougui/Deminer/logs . && awkfields -F 'loss' -f logs/deminer.log | plot --xlabel 'step' --ylabel 'loss'  --ws 100
