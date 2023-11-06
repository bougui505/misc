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
    bar [-f FIELD] [-s SCALE]
EOF
    exit
}

SCALE=1
while getopts ':h:f:s:' opt; do
    case $opt in
        (f) FIELD=$OPTARG;;
        (s) SCALE=$OPTARG;;
        (h) usage;;
        (*) usage;;
    esac
done

MAXLEN=100 # Maximum length of the bar

cat /dev/stdin \
    | awk -v FIELD=$FIELD -v MAXLEN=$MAXLEN -v SCALE=$SCALE '{
    printf $0"\t"
    LENGTH=$FIELD*SCALE
    if (LENGTH > MAXLEN){
        LENGTH=MAXLEN
    }
    for (i=1;i<=LENGTH;i++){
        printf "#"
    }
    if (LENGTH >= MAXLEN){
        printf "..."
    }
    printf "\n"
}' \
    | column -t
