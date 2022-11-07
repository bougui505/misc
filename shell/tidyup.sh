#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/

# Tidy up files by date. Create a directory for each date based on modification
# time and put files into it

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection
setopt extendedglob

function usage () {
    cat << EOF
Help message
Create and move files and directories based on their timestamps.
    -d, --dir name of the base directory to put the data in (default: daily)
    -o, --older only treat files older than the given elapsed timelapse in hour (default: 24)
    -h, --help print this help message and exit
EOF
}

DIR='daily'
OLDER=24
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dir) DIR="$2"; shift ;;
        -o|--older) OLDER="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

rename -v 's/ /_/g' *
rename -v 's/\(/_/g' *
rename -v 's/\)/_/g' *

if [ ! -z $DIR ]; then
    [[ ! -d $DIR ]] && mkdir -v $DIR
else
    DIR=""
fi
STAT=$(stat -c '%y %Y %n' *~$DIR)
echo $STAT \
    | awk '{print gensub("-","","g",$1),$4,$5}' \
    | awk -v older=$OLDER -v now=$(date +%s) '
    BEGIN{older=older*60*60}
    {
    timestamp=$1
    epoch=$2
    fname=$3
    delta=now-epoch
    if (delta>older){
        print $0}
    }
    ' \
    | while read LINE; do
        CREATEDIR="$DIR/$(echo $LINE | awk '{print $1}')"
        FILETOMOVE="$(echo $LINE | awk '{print $3}')"
        [[ ! -d $CREATEDIR ]] && mkdir -v $CREATEDIR
        mv -v $FILETOMOVE $CREATEDIR
    done
