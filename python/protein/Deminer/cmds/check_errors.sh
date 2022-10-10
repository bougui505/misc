#!/usr/bin/env zsh
rsync desk:/ld18-1006/work/bougui/Deminer/logs . && grep -i error logs/deminer.log
