#!/usr/bin/env zsh
rsync desk:/ld18-1006/work/bougui/Deminer/logs . &&
LASTUPD=$(grep ': epoch:' logs/deminer.log | tail -1 | awk '{print $1,$2}') &&
RESTARTPOS=$(grep "Starting /ld18-1006/work/bougui/Deminer/deminer.py\|step:" logs/deminer.log | uniq | cut -c 26- | uniq | grep -n 'Starting' | awk -F: '{printf "--vline "$1" "}')
awkfields -F 'loss' -f logs/deminer.log | plot --xlabel 'step' --ylabel 'loss'  --ws 1000 --xmin 0 --ymax 2.6 --grid --title $LASTUPD --save /home/bougui/Documents/presentations/20220929_DeMiner/figs/loss.svg $(echo "$RESTARTPOS")
