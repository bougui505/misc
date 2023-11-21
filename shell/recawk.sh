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

bold=$(tput bold)
normal=$(tput sgr0)

function usage () {
    cat << EOF

    -h, --help print this help message and exit

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

An ${bold}END${normal} can be given as in standard awk to run a command when awk has parsed the full file(s).

    ${bold}zcat data/file.rec.gz | recawk '{a[nr]=rec["i"]}END{for (i in a){print i, a[i]}}'${normal}

${bold}-v${normal} can be given as in standard awk command. E.g. ${bold}recawk -v "A=1" '{...}'${normal}

    ${bold}zcat data/file.rec.gz | recawk -v "ania=ciao" '{printrec();print("ania="ania);print("--")}'${normal}

Examples:
    
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
case $1 in
    -h|--help) usage; exit 0 ;;
    -v) shift; V=$1; shift ;;
esac

if [ "$#" -eq 0 ]; then
    usage; exit 0
fi

CMD=$(echo $1 | awk -F"END" '{print $1}')
ENDCMD=$(echo $1 | awk -F"END" '{print $2}')
FILENAMES="${@:2}"

echo $V
awk -v $V -F"=" '
function printrec(){
    for (field in rec){
        print field"="rec[field]
    }
}
BEGIN{
nr=0
}
{
if ($0=="--"){
    nr+=1
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
