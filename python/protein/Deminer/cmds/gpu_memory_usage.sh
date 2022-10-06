#!/usr/bin/env zsh
rsync desk:/ld18-1006/work/bougui/Deminer/logs . && cat logs/deminer.log|awkfields -F gpu_memory_usage|tr -d '%'| plot -H --xlabel 'GPU memory usage (%)' --ylabel 'count'
