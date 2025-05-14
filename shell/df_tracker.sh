#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed May 14 10:17:22 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

OUTFILE="$HOME/.df_tracker.txt"
MAXLINES=100000

# print help if --help is passed
function usage() {
    echo "Usage: $0 [--help]"
    echo "Track disk usage of the root filesystem."
    echo "Outputs the data to $OUTFILE."
    echo "Options:"
    echo "  --help, -h  Show this help message"
    echo "  --plot, -p  Plot the data"
}
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "$1" == "--plot" || "$1" == "-p" ]]; then
    YMIN=$(awk 'NR>1 {print $5/(1024^2)}' "$OUTFILE" | sort -n | head -n 1)
    YMAX=$(awk 'NR>1 {print $5/(1024^2)}' "$OUTFILE" | sort -n | tail -n 1)
    YMAX=$(echo "scale=2; $YMAX+($YMAX-$YMIN)/10" | bc)
    YMIN=$(echo "scale=2; $YMIN-($YMAX-$YMIN)/10" | bc)
    cat "$OUTFILE" \
        | awk 'NR>1 {print $1,$3/(1024^2),$5/(1024^2)}' \
        | plot3 --xlabel "date" \
                --ylabel "disk available (GB)" \
                plot \
                    --fields "ts y y" \
                    --labels "total available" \
                    --ymin $YMIN \
                    --ymax $YMAX \
                    --shade "1 1" \
                    --alpha-shade 1 \
                    --fmt "lightcoral lightcyan"
    exit 0
fi

HEADER="timestamp filesystem size used available use% mounted_on"
if [ ! -f "$OUTFILE" ]; then
    echo "$HEADER" > "$OUTFILE"
fi
SECONDS_SINCE_EPOCH=$(date +%s)
DF=$(df / | awk 'NR>1')
echo "$SECONDS_SINCE_EPOCH $DF" >> "$OUTFILE"

# Remove lines if the file exceeds MAXLINES
if [ $(wc -l < "$OUTFILE") -gt $MAXLINES ]; then
    tail -n $MAXLINES "$OUTFILE" > "${OUTFILE}.tmp"
    rm "$OUTFILE"
    echo "$HEADER" > "$OUTFILE"
    cat "${OUTFILE}.tmp" >> "$OUTFILE"
    rm "${OUTFILE}.tmp"
fi
