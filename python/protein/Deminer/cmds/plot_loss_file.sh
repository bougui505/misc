#!/usr/bin/env zsh
rsync desk:/ld18-1006/work/bougui/Deminer/logs . &&
LASTUPD=$(grep ': epoch:' logs/deminer.log | tail -1 | awk '{print $1,$2}') &&
awkfields -F 'loss' -f logs/deminer.log | plot --xlabel 'step' --ylabel 'loss'  --ws 1000 --xmin 0 --ymax 2.6 --grid --title $LASTUPD --save /home/bougui/Documents/presentations/20220929_DeMiner/figs/loss.svg
