#!/usr/bin/env bash

Cyan='\033[0;36m'
NC='\033[0m' # No Color

echo -e "$Cyan"
cat << EOF
$(date): sourcing $0
EOF
echo -e $FUNFILES$NC

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap '/bin/rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

[ -z $FUNFILES ] && export FUNFILES=""
precmd() {
    [ -f fun.sh ] && source fun.sh && FUNFILES+="$(realpath fun.sh)" && FUNFILES=$(echo $FUNFILES | awk -F":" '{for (i=1;i<=NF;i++){print $i}}' | sort -u | awk '{printf $1":"}')
}

test1 () {
    ../awkpaste.sh =(seq 10) =(seq 13)
}

test2 () {
    ../awkpaste.sh =(seq 13) =(seq 10)
}

test3 () {
    ../awkpaste.sh =(seq 13) =(seq 10) =(seq 15)
}

test4 () {
    paste -d" " =(seq 14) =(seq 10) >! /tmp/a
    paste -d" " =(seq 20 30) =(seq 50 70) >! /tmp/b
    ../awkpaste.sh /tmp/a /tmp/b
}
