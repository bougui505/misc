#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jun 27 16:50:37 2025

# Default values
DEFAULT_COUNT=10
INCLUDE_HIDDEN=0
SHOW_TIME=0
SORT_BY_MODIFICATION=0 # Added to track -m option
LAST_24H=0             # Added to track -D option

# --- Function to display script usage ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Lists the most recently accessed or modified files recursively in the current directory."
    echo ""
    echo "Options:"
    echo "  -n <count>   Set the number of top files to display (default: $DEFAULT_COUNT)."
    echo "  -a           Include hidden files and directories (starting with '.')."
    echo "  -m           Sort by MODIFICATION Time (%y) instead of Access Time (%x)."
    echo "  -t           Display the Access/Modification Time along with the filename."
    echo "  -D           List files accessed in the last 24 hours. Implies sorting by access time."
    echo "  -h           Display this help message."
    exit 1
}

# --- Parse arguments ---
while getopts "n:amthD" opt; do
    case ${opt} in
        n )
            if ! [[ "$OPTARG" =~ ^[0-9]+$ ]] || [ "$OPTARG" -le 0 ]; then
                echo "Error: -n value must be a positive integer." >&2
                usage
            fi
            COUNT=$OPTARG
            ;;
        a )
            INCLUDE_HIDDEN=1
            ;;
        m )
            SORT_BY_MODIFICATION=1 # Set flag when -m is present
            ;;
        t )
            SHOW_TIME=1
            ;;
        D )
            LAST_24H=1
            ;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# Set default count if -n was not provided
COUNT=${COUNT:-$DEFAULT_COUNT}

# If -D is used, force sorting by access time
if [ "$LAST_24H" -eq 1 ]; then
    SORT_BY_MODIFICATION=0
fi

# --- Construct the stat --printf format string ---
# This will always output the sorting key (epoch time) first,
# optionally followed by the human-readable time, and then the filename.
STAT_PRINTF_FORMAT=""

if [ "$SORT_BY_MODIFICATION" -eq 1 ]; then
    STAT_PRINTF_FORMAT='%Y' # Modification time (Epoch seconds) for sorting
    DISPLAY_TIME_FORMAT_PART='%y' # Corresponding human-readable modification time
else
    STAT_PRINTF_FORMAT='%X' # Access time (Epoch seconds) for sorting
    DISPLAY_TIME_FORMAT_PART='%x' # Corresponding human-readable access time
fi

if [ "$SHOW_TIME" -eq 1 ]; then
    STAT_PRINTF_FORMAT="$STAT_PRINTF_FORMAT $DISPLAY_TIME_FORMAT_PART"
fi

STAT_PRINTF_FORMAT="$STAT_PRINTF_FORMAT %n\0"

# --- 1. Construct the 'find' Command ---
FIND_TIME_FILTER=""
if [ "$LAST_24H" -eq 1 ]; then
    FIND_TIME_FILTER="-atime -1"
fi

if [ "$INCLUDE_HIDDEN" -eq 0 ]; then
    # Exclude hidden files
    FIND_CMD="find . -path '*/.*' -prune -o -type f $FIND_TIME_FILTER -exec stat --printf '$STAT_PRINTF_FORMAT' {} +"
else
    # Include hidden files
    FIND_CMD="find . -type f $FIND_TIME_FILTER -exec stat --printf '$STAT_PRINTF_FORMAT' {} +"
fi

# --- 2. Execute the entire pipeline directly ---

# Pipe the find output directly to sort
if [ "$SHOW_TIME" -eq 0 ]; then
    # Goal: Filename only. Needs cut.
    # $FIND_CMD -> sort -znr -> cut -z -> tr -> head
    eval "$FIND_CMD" | \
        sort -znr | \
        cut -z -f2- -d' ' | \
        tr '\0' '\n' | \
        head -n "$COUNT"
else
    # Goal: Timestamp and Filename. Skip cut.
    # $FIND_CMD -> sort -znr -> tr -> head
    eval "$FIND_CMD" | \
        sort -znr | \
        tr '\0' '\n' | \
        head -n "$COUNT"
fi
