#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb 15 13:08:28 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Read a list of timestamps in seconds since Epoch and compute the progression from it
For example to compute the timestamps for a list of files use:
    
    `stat -c %Y file1 file2 ...`

    -h, --help print this help message and exit
    -n, --number expected number of elements in the list
EOF
}

N=0  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

sort -g \
| awk -v n=$N '
BEGIN {"date" |  getline mydate}
{
if (NR==1){
    t0=$1
}
tf=$1
}
END{
print("date="mydate)
rate = NR/(tf-t0)
print("rate="rate" s⁻¹")
if (n>0){
    progress = NR*100/n
    print("progress="progress" %")
    eta = (n-NR)/rate
    "qalc -t "eta" s" | getline eta
    print("eta="eta)
    print("--")
}
}' > $MYTMP/1708003112.rec

BAR=$(cat $MYTMP/1708003112.rec \
    | recawk '{print rec["progress"]}' \
    | bar -s 0.5)

cat $MYTMP/1708003112.rec | sed '/^--$/d'
echo "bar=$BAR"
echo "--"
