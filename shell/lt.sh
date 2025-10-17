#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2018-02-06 10:21:20 (UTC+0100)

# Filter files by modification time

usage ()
{
    echo "Usage"
    echo "lt"
    echo "Default: list files from the oldest file to the newest file"
    echo "-l: lower timestamp as given in column 6 of the output of lt
              e.g.: '2017-11-07 09:08:54'"
    echo "-u: upper timestamp as given in column 6 of the output of lt
              e.g. '2017-11-07 09:11:26'"
}

underline_last_modified () {
	# Function to underline files changes:
	# last 2 min (120s) -> red
	# last 5 min (300s) -> light red
	# last 10 min (600s) -> blue
	# Cluster also by 12 hours (43200s)
	exa --icons -lh -snew --git --time-style full-iso --links --color=always $@ | awk '{Time=$5" "$6
			gsub("\x1B\\[[0-9;]*[a-zA-Z]","",Time)
			gsub("[-,:]", " ", Time)
			Time=mktime(Time)
			delta=systime()-Time
			if (delta<120){
				gsub("\x1B\\[[0-9;]*[a-zA-Z]","", $0)
				print "\033[31m"$0
			}
			else if (delta>=120 && delta<300){
				gsub("\x1B\\[[0-9;]*[a-zA-Z]","", $0)
				print "\033[91m"$0
			}
			else if (delta>=300 && delta<600){
				gsub("\x1B\\[[0-9;]*[a-zA-Z]","", $0)
				print "\033[34m"$0
			}
			else if (Time-Time_prev>43200){
				print ""
				print $0
			}
			else {print $0}
			Time_prev = Time
			}'
	# See color formats here: https://misc.flogisoft.com/bash/tip_colors_and_formatting
	}

LOWER=None
UPPER=None
for i in "$@"; do
    case $i in
        "-h")
            usage
            exit
            ;;
        "-l")
            shift
            LOWER=$1
            shift
            ;;
        "-u")
            shift
            UPPER=$1
            shift
    esac
done
FILENAME=$@
if [ "$LOWER" = "None" ] && [ "$UPPER" = "None" ]; then
    if hash exa; then
        if exa --help | grep -q -- --git; then
            underline_last_modified $(echo $FILENAME)
        else
            # exa exists but does not support --git, fall back to ls
            ls -rlth --time-style=+"%F %H:%M:%S.%N" --color $(echo $FILENAME)
        fi
    else
        # exa does not exist, fall back to ls
        ls -rlth --time-style=+"%F %H:%M:%S.%N" --color $(echo $FILENAME)
    fi
    exit
fi
if [ "$LOWER" = "None" ]; then
    LOWER="1970-01-01 01:00:00"
fi
if [ "$UPPER" = "None" ]; then
    UPPER=$(date)
fi
LOWER=$(date +%s.%N -d"$LOWER")
UPPER=$(date +%s.%N -d"$UPPER")
#echo $LOWER $UPPER
ls -rlth --time-style=+%s.%N $(echo $FILENAME) | awk -v LOWER=$LOWER -v UPPER=$UPPER '{if ($6>=LOWER && $6<=UPPER){print $7}}'
