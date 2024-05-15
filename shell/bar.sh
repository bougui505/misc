#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-01-10 16:20:44 (UTC+0100)

# Simple bar chart plotting in a terminal by reading integer from stdin

# FIELD: Field number to read the value from

usage () {
    cat << EOF
Usage:
    bar [-f FIELD] [-s SCALE] [-n NUM]

        -f field number to consider
        -s scale of the bar
        -n maximum number (useful for progress bar)
EOF
    exit
}

SCALE=1
MAXLEN=100 # Maximum length of the bar
while getopts ':h:f:s:n:' opt; do
    case $opt in
        (f) FIELD=$OPTARG;;
        (s) SCALE=$OPTARG;;
        (n) MAXLEN=$OPTARG;;
        (h) usage;;
        (*) usage;;
    esac
done

cat /dev/stdin \
    | awk -v FIELD=$FIELD -v MAXLEN=$MAXLEN -v SCALE=$SCALE '{
    printf $0"\t"
    LENGTH=$FIELD*SCALE
    MAXLEN=MAXLEN
    for (i=1;i<=MAXLEN+1;i++){
        if (i<=LENGTH){
            printf "#"
        }
        else if (i<=MAXLEN){
            printf "."
        }
        else{
            printf "|"
        }
    }
    if (LENGTH > MAXLEN){
        printf "..."
    }
    printf "\n"
}' \
    | column -t
