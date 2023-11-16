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
    zcat data/file.rec.gz | recawk '{print rec["i"]}'
}

test2 () {
    zcat data/file.rec.gz | recawk '{for (field in rec){print field}}'
}

test3 () {
    zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}'
}

test4 () {
    zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("NR="NR);print("--")}'
}

test5 () {
    zcat data/file.rec.gz | recawk '{
        a[nr]=rec["i"]
    }
    END{
    for (i in a){
        print i, a[i]
    }}'
}
