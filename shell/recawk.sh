#!/usr/bin/env bash
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

bold=$(tput bold)
normal=$(tput sgr0)

function usage () {
    cat << EOF

    -h, --help print this help message and exit
    -n, --nrec print the number of records for the given rec file
    -s, --sample give the number of records to pick up randomly from the given rec file

----------------${bold}RECAWK${normal}----------------

Read a rec file formatted as:

${bold} 
key1=val1
key2=val2
--
key1=val12
key2=val22
--
[...]
${normal} 

using awk.

An example rec file can be found in ${bold}$DIRSCRIPT/recawk_test/data/file.rec.gz${normal}

The key, value couples are stored in ${bold}rec${normal} awk array.
To access key1, use:

    ${bold}rec["key1"]${normal} -> val1 (if in first records)

    ${bold}zcat data/file.rec.gz | recawk '{print rec["i"]}'${normal}

The full rec file is not stored in ${bold}rec${normal}. Just the current record is stored.

To enumerate fields just use:

    ${bold} 
    for (field in rec){
        print field
    }
    ${normal} 

A function ${bold}printrec()${normal} can be used to print the current record. The record separator "--" is not printed by ${bold}printrec()${normal} to allow the user to add an item to the record:

    ${bold}zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}'${normal}

Variable ${bold}nr${normal} is defined. ${bold}nr${normal} is the number of input records awk has processed since the beginning of the program’s execution. Not to be confused with ${bold}NR${normal}, which is the builtin awk variable, which store the number of rows/lines awk has processed since the beginning of the program’s execution.

    ${bold}zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("NR="NR);print("--")}'${normal}

Variable ${bold}fnr${normal} is defined. ${bold}fnr${normal} is the number of input records awk has processed for the current file. Not to be confused with ${bold}FNR${normal}, which is the builtin awk variable, which store the number of rows/lines awk has processed for the current file.

    ${bold}recawk '{print NR,FNR,nr,fnr}' =(zcat data/file.rec.gz) =(zcat data/file.rec.gz)${normal}

An ${bold}END${normal} can be given as in standard awk to run a command when awk has parsed the full file(s).

    ${bold}zcat data/file.rec.gz | recawk '{a[nr]=rec["i"]}END{for (i in a){print i, a[i]}}'${normal}

${bold}-v${normal} can be given as in standard awk command. E.g. ${bold}recawk -v "A=1" '{...}'${normal}

    ${bold}zcat data/file.rec.gz | recawk -v "ania=ciao" '{printrec();print("ania="ania);print("--")}'${normal}

The semicolon ";" terminates the statement. It is highly recommanded to put the semicolon ";" at the end of the statements, even in a script on multiple lines, to avoid bugs.

${bold}IMPORTANT REMARKS${normal}

- For ${bold}float or integer values${normal}, string ${bold}conversion to float or integer${normal} is needed using;
    '{value=rec["key"]*1}'

${bold}EXAMPLES${normal}
${bold} 
    zcat data/file.rec.gz | recawk '{print rec["i"]}'
    zcat data/file.rec.gz | recawk '{for (field in rec){print field}}'
    zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}
    zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("NR="NR);print("--")}'
    zcat data/file.rec.gz | recawk '{a[nr]=rec["i"]}END{for (i in a){print i, a[i]}}'
    zcat data/file.rec.gz | recawk -v "ania=ciao" '{printrec();print("ania="ania);print("--")}'
${normal} 

EOF
}

V="V=0"
GETNREC=0
SAMPLE=0
case $1 in
    -h|--help) usage; exit 0 ;;
    -v) shift; V=$1; shift ;;
    -n|--nrec) GETNREC=1 ;;
    -s|--sample) SAMPLE=$2; shift ;;
esac

getnrec(){
    grep -c "^--$" $1
}

if [ "$#" -eq 0 ]; then
    usage; exit 0
fi

CMD=$(echo "$1" | tr "\n" "$" | awk -F"END" '{print $1}' | tr "$" "\n")
ENDCMD=$(echo "$1" | tr "\n" "$" | awk -F"END" '{print $2}' | tr "$" "\n")
FILENAMES="${@:2}"

if [[ $GETNREC -eq 1 ]]; then
    getnrec $FILENAMES
    exit 0
fi

if [[ $SAMPLE -gt 0 ]]; then
    cat > $MYTMP/in
    FILENAMES="$MYTMP/in $FILENAMES"
    NREC=$(getnrec $FILENAMES)
    V="NREC=$NREC"
    if [[ $SAMPLE -lt NREC ]]; then
        CMD='{if (fnr in RECSEL){printrec();print("--")}}'
    else
        cat $MYTMP/in
        exit 0
    fi
fi

awk -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" '
function printrec(){
    for (field in rec){
        print field"="rec[field]
    }
}
BEGIN{
srand(seed)
nr=0
if (SAMPLE>0){
    n=0
    while (n<SAMPLE){
        r=int(NREC*rand())
        RECSEL[r]=r
        n=length(RECSEL)
    }
}
}
{
if (FNR==1){
    fnr=0
}
if ($0=="--"){
    nr+=1
    fnr+=1
    '"$CMD"'
    delete rec
}
else{
    rec[$1]=substr($0,length($1)+2)
}
}
END{
    '"$ENDCMD"'
}
' $FILENAMES
