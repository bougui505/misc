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
SHOW_TIME=0      # Default to only showing the filename

# --- Function to display script usage ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Lists the most recently accessed or modified files recursively in the current directory."
    echo ""
    echo "Options:"
    echo "  -n <count>   Set the number of top files to display (default: $DEFAULT_COUNT)."
    echo "  -a           Include hidden files and directories (starting with '.')."
    echo "  -m           Sort by MODIFICATION Time (%Y) instead of Access Time (%X)."
    echo "  -t           Display the Access/Modification Time along with the filename."
    echo "  -h           Display this help message."
    exit 1
}

# --- Parse arguments ---
while getopts "n:amth" opt; do
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
            TIME_FORMAT='%Y'
            ;;
        t )
            SHOW_TIME=1
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

if [ "$SHOW_TIME" -eq 0 ]; then
    TIME_FORMAT='%X' # Default to Access Time (Epoch seconds)
else
    TIME_FORMAT='%x'
fi

# --- 1. Construct the 'find' Command ---

if [ "$INCLUDE_HIDDEN" -eq 0 ]; then
    # Exclude hidden files
    FIND_CMD="find . -path '*/.*' -prune -o -type f -exec stat --printf '$TIME_FORMAT %n\0' {} +"
else
    # Include hidden files
    FIND_CMD="find . -type f -exec stat --printf '$TIME_FORMAT %n\0' {} +"
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
