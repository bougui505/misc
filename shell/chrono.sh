#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Apr 15 14:34:38 2025

# set -e  # exit on error
# set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script
trap print_totaltime EXIT

function usage () {
    cat << EOF
Usage: cmd | $(basename "$0") [options]
The script will read lines from standard input and print the elapsed time in the format HH:MM:SS.mmm.
The elapsed time is printed in green, orange or red depending on the time elapsed,
green for less than MEDIUM, orange for between MEDIUM and HIGH, and red for more than HIGH.
The total running time will be printed in cyan and underlined when the input is closed.
The script will exit when the input is closed or when the user presses Ctrl+C.
Options:
    -h, --help print this help message and exit
    -e, --elapsed print elapsed time since the start of the script
    -m, --medium <number><ms|s|m|h> set the medium time when the time is printed in orange (ms for milliseconds, s for seconds, m for minutes, h for hours)
        default: 1m
    -d, --duration <number><ms|s|m|h> set the estimated duration of the script in (ms for milliseconds, s for seconds, m for minutes, h for hours) to calculate estimated progress
        default: 0s (unknown)
    --eta print the estimated time of arrival (ETA). -d must be set to a value greater than 0s
    -H, --high <number><ms|s|m|h> set the high time when the time is printed in red (ms for milliseconds, s for seconds, m for minutes, h for hours)
        default: 5m
EOF
}

ELAPSED=0  # Default value
MEDIUM=1m  # medium time when the time is printed in orange (ms for milliseconds, s for seconds, m for minutes, h for hours)
HIGH=5m  # high time when the time is printed in red (ms for milliseconds, s for seconds, m for minutes, h for hours)
DURATION=0s  # Estimated duration of the script in seconds to calculate progess, default is 0s (unknown)
ETA=0  # Boolean to switch on the ETA calculation
while [ "$#" -gt 0 ]; do
    case $1 in
        -e|--elapsed) ELAPSED=1 ;;
        -m|--medium) shift; MEDIUM="$1" ;;
        -H|--high) shift; HIGH="$1" ;;
        -d|--duration) shift; DURATION="$1" ;;
        --eta) ETA=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

# Convert MEDIUM to milliseconds
if grep -q 'ms' <<< "$MEDIUM"; then
    MEDIUM=${MEDIUM//ms/}
elif [[ $MEDIUM =~ ^([0-9]+)([smh])$ ]]; then
    case ${BASH_REMATCH[2]} in
        s) MEDIUM=$(( ${BASH_REMATCH[1]} * 1000 )) ;;
        m) MEDIUM=$(( ${BASH_REMATCH[1]} * 60000 )) ;;
        h) MEDIUM=$(( ${BASH_REMATCH[1]} * 3600000 )) ;;
    esac
else
    echo "Error: MEDIUM is not in the format <number><s|m|h>" >&2
    exit 1
fi
# Convert HIGH to milliseconds
if grep -q 'ms' <<< "$HIGH"; then
    HIGH=${HIGH//ms/}
elif [[ $HIGH =~ ^([0-9]+)([smh])$ ]]; then
    case ${BASH_REMATCH[2]} in
        s) HIGH=$(( ${BASH_REMATCH[1]} * 1000 )) ;;
        m) HIGH=$(( ${BASH_REMATCH[1]} * 60000 )) ;;
        h) HIGH=$(( ${BASH_REMATCH[1]} * 3600000 )) ;;
    esac
else
    echo "Error: HIGH is not in the format <number><s|m|h>" >&2
    exit 1
fi
# Convert DURATION to milliseconds
if grep -q 'ms' <<< "$DURATION"; then
    DURATION=${DURATION//ms/}
elif [[ $DURATION =~ ^([0-9]+)([smh])$ ]]; then
    case ${BASH_REMATCH[2]} in
        s) DURATION=$(( ${BASH_REMATCH[1]} * 1000 )) ;;
        m) DURATION=$(( ${BASH_REMATCH[1]} * 60000 )) ;;
        h) DURATION=$(( ${BASH_REMATCH[1]} * 3600000 )) ;;
    esac
else
    echo "Error: DURATION is not in the format <number><s|m|h>" >&2
    exit 1
fi

function print_line () {
    sleep 0.001s  # to avoid missing letters in the output
    echo "$deltat $line"
    if [[ $ELAPSED -eq 0 ]]; then 
        t0=$(date +%s%3N)
    fi
}

function print_totaltime () {
    # format TOTALTIME in HH:MM:SS.mmm
    TOTALTIME=$(printf "%02d:%02d:%02d.%03d" $((TOTALTIME/3600000)) $(( (TOTALTIME%3600000)/60000 )) $(( (TOTALTIME%60000)/1000 )) $(( TOTALTIME%1000 )))
    # echo the total time in cyan and underlined
    TOTALTIME=$(echo -ne "\033[0;36m\033[4m$TOTALTIME\033[0m")
    echo -ne "\nTotal time: $TOTALTIME"
}

# Get start time in milliseconds
T0=$(date +%s%3N)  # Global start time of the script
t0=$(date +%s%3N)
while sleep 0.001s; do
    t1=$(date +%s%3N)
    deltat_ms=$((t1 - t0))
    deltat=$(printf "%02d:%02d:%02d.%03d" $((deltat_ms/3600000)) $(( (deltat_ms%3600000)/60000 )) $(( (deltat_ms%60000)/1000 )) $(( deltat_ms%1000 )))
    TOTALTIME=$(($t1 - $T0))
    if [[ DURATION -gt 0 ]]; then
        # Calculate the progress
        progress="$((100 * TOTALTIME / DURATION))%"
        # format progress with a fixed width of 4 characters
        progress=$(printf "%4s" "$progress")
        deltat="$deltat $progress"
    fi
    if [[ ETA -eq 1 && DURATION -gt 0 ]]; then
        # Calculate the ETA
        if [[ $TOTALTIME -gt 0 ]]; then
            eta=$((DURATION - TOTALTIME))
            eta=$(printf "%02d:%02d:%02d.%03d" $((eta/3600000)) $(( (eta%3600000)/60000 )) $(( (eta%60000)/1000 )) $(( eta%1000 )))
            deltat="$deltat ETA:$eta"
        fi
    fi
    if [[ $deltat_ms -gt $MEDIUM && $deltat_ms -le $HIGH ]]; then
        # format deltat in orange
        deltat=$(echo -ne "\033[0;33m$deltat\033[0m")
    elif [[ $deltat_ms -gt $HIGH ]]; then
        # format deltat in red
        deltat=$(echo -ne "\033[0;31m$deltat\033[0m")
    else
        # format deltat in green
        deltat=$(echo -ne "\033[0;32m$deltat\033[0m")
    fi
    read -r -t 0.001 line
    ret=$?
    if [[ $ret -eq 1 ]]; then  # this is the end of the input
        # print_totaltime  # not needed, the trap will do it
        exit 0
    elif [[ $ret -eq 0 ]]; then  # a line was read
        print_line
    else  # print the elapsed time
        echo -ne "$deltat\r"
    fi
done < "${1:-/dev/stdin}"
