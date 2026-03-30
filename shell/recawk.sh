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

    -h, --help          print this help message and exit
    -n, --nrec          print the number of records for the given rec file
    -s, --sample        give the number of records to pick up randomly from the given rec file
    -k, --keys          print all keys present in the file
    --torec <separator> convert a column file with the first line as keys and the rest
                        of the lines as values to a rec file. Columns are separated by <separator>.
                        The first line is used as keys, the rest of the lines as values.
                        The output is written to stdout.
    --tocsv             convert a rec file to csv format on stdout. The first line contains the keys
                        and the following lines contain the values for each record.

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

Variables ${bold}nr${normal} and ${bold}fnr${normal} are defined:
- ${bold}nr${normal}: number of input records awk has processed since the beginning of the program's execution
- ${bold}fnr${normal}: number of input records awk has processed for the current file

    ${bold}zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("fnr="fnr);print("--")}'${normal}

An ${bold}END${normal} can be given as in standard awk to run a command when awk has parsed the full file(s).

    ${bold}zcat data/file.rec.gz | recawk '{a[nr]=rec["i"]}END{for (i in a){print i, a[i]}}'${normal}

${bold}-v${normal} can be given as in standard awk command. E.g. ${bold}recawk -v "A=1" '{. ..}'${normal}

    ${bold}zcat data/file.rec.gz | recawk -v "ania=ciao" '{printrec();print("ania="ania);print("--")}'${normal}

The semicolon ";" terminates the statement. It is highly recommanded to put the semicolon ";" at the end of the statements, even in a script on multiple lines, to avoid bugs.

${bold}IMPORTANT REMARKS${normal}

- For ${bold}float or integer values${normal}, string ${bold}conversion to float or integer${normal} is needed using;
    '{value=rec["key"]*1}'

${bold}SPEARMAN FUNCTION${normal}

A ${bold}spearman(x, y, n)${normal} function is available to compute Spearman correlation coefficient between two arrays x and y of length n.

${bold}PEARSON FUNCTION${normal}

A ${bold}pearson(x, y, n)${normal} function is available to compute Pearson correlation coefficient between two arrays x and y of length n.

${bold}EXAMPLES${normal}
${bold}
    # Compute Spearman correlation between two arrays
    zcat data/file.rec.gz | recawk '{x[nr]=rec["x"]; y[nr]=rec["y"]}END{print spearman(x, y, nr)}'
    
    # Compute Pearson correlation between two arrays
    zcat data/file.rec.gz | recawk '{x[nr]=rec["x"]; y[nr]=rec["y"]}END{print pearson(x, y, nr)}'
    
    # Print the value of key 'i' from each record
    zcat data/file.rec.gz | recawk '{print rec["i"]}'
    
    # Print all keys in the file
    zcat data/file.rec.gz | recawk '{for (field in rec){print field}}'
    
    # Print each record with an additional key-value pair
    zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}'
    
    # Print record number and file record number
    zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("fnr="fnr);print("--")}'
    
    # Store all values of key 'i' in an array
    zcat data/file.rec.gz | recawk '{a[nr]=rec["i"]}END{for (i in a){print i, a[i]}}'
    
    # Use a custom variable
    zcat data/file.rec.gz | recawk -v "ania=ciao" '{printrec();print("ania="ania);print("--")}'
    
    # Print all keys in the file
    zcat data/file.rec.gz | recawk --keys
    
    # Convert a CSV file to rec format
    cat data.csv | recawk --torec ","
${normal}

EOF
}

V="V=0"
GETNREC=0
SAMPLE=0
TOREC=0
KEYS=0
TOCSV=0
case $1 in
    -h|--help) usage; exit 0 ;;
    -v) shift; V=$1; shift ;;
    -n|--nrec) GETNREC=1 ;;
    -s|--sample) SAMPLE=$2; shift ;;
    --torec) TOREC=$2; shift ;;
    --keys) KEYS=1 ;;
    --tocsv) TOCSV=1 ;;
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

if [[ $KEYS -eq 1 ]]; then
    awk -F"=" '{
        if (FNR==1){
            fnr=0
        }
        if ($0=="--"){
            fnr+=1
        }
        else{
            keys[$1]=1
        }
    }
    END{
        for (k in keys){
            print k
        }
    }' $FILENAMES
    exit 0
fi

if [[ $TOCSV -eq 1 ]]; then
    awk '
    BEGIN {
        first = 1
    }
    {
        if (FNR == 1) {
            fnr = 0
        }
        if ($0 == "--") {
            if (first) {
                # Print header
                for (i = 1; i <= nkeys; i++) {
                    printf "%s", keys[i]
                    if (i < nkeys) printf ","
                }
                printf "\n"
                first = 0
            }
            # Print values for current record
            for (i = 1; i <= nkeys; i++) {
                key = keys[i]
                printf "%s", rec[key]
                if (i < nkeys) printf ","
            }
            printf "\n"
            delete rec
        } else {
            # Store key-value pairs
            split($0, a, "=")
            key = a[1]
            value = a[2]
            rec[key] = value
            # Add key to keys array if not already present
            found = 0
            for (i = 1; i <= nkeys; i++) {
                if (keys[i] == key) {
                    found = 1
                    break
                }
            }
            if (!found) {
                keys[++nkeys] = key
            }
        }
    }
    ' "$FILENAMES"
    exit 0
fi

if [[ $TOREC != 0 ]]; then
    if [[ -z $TOREC ]]; then
        echo "Error: --torec requires a separator argument." >&2
        exit 1
    fi
    if [[ $TOREC == " " ]]; then
        awk -v FPAT="[^[:space:]]+|(\"([^\"]|\"\")*\")" '{
            if (NR==1){
                for (i=1; i<=NF; i++){
                    # gsub(/ /, "_", $i)
                    keys[i]=$i
                }
            }
            else{
                for (i=1; i<=NF; i++){
                    if (keys[i] != ""){
                        print keys[i]"="$i
                    }
                }
                print "--"
            }
        }' "$FILENAMES"
    else
        awk -v FPAT="[^$TOREC]*|(\"([^\"]|\"\")*\")" '{
            if (NR==1){
                for (i=1; i<=NF; i++){
                    # gsub(/ /, "_", $i)
                    keys[i]=$i
                }
            }
            else{
                for (i=1; i<=NF; i++){
                    if (keys[i] != ""){
                        print keys[i]"="$i
                    }
                }
                print "--"
            }
        }' "$FILENAMES"
    fi
    exit 0
fi

if [[ $SAMPLE -gt 0 ]]; then
    # For sampling, we need to count records first to determine if we need to sample
    # When data comes from a pipe, we must read it all into a temporary file first
    # Check if input is from a pipe
    if [ ! -t 0 ]; then
        # Input is from a pipe, read all records into temporary file
        cat > $MYTMP/in
        FILENAMES="$MYTMP/in $FILENAMES"
    fi
    
    # Count records
    NREC=$(awk 'BEGIN{nr=0} $0=="--"{nr++} END{print nr}' "$FILENAMES")
    V="NREC=$NREC"
    if [[ $SAMPLE -lt NREC ]]; then
        CMD='{if (fnr in RECSEL){printrec();print("--")}}'
    else
        # If sample size is greater than or equal to total records, just print all
        CMD='{printrec();print("--")}'
        exit 0
    fi
fi

awk -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" '
function printrec(){
    for (field in rec){
        print field"="rec[field]
    }
}

function spearman(x, y, n) {
    # Create arrays for ranking
    delete rank_x
    delete rank_y
    delete sorted_x
    delete sorted_y
    
    # Copy arrays
    for (i = 1; i <= n; i++) {
        sorted_x[i] = x[i]
        sorted_y[i] = y[i]
    }
    
    # Sort arrays
    for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
            if (sorted_x[i] > sorted_x[j]) {
                temp = sorted_x[i]
                sorted_x[i] = sorted_x[j]
                sorted_x[j] = temp
            }
            if (sorted_y[i] > sorted_y[j]) {
                temp = sorted_y[i]
                sorted_y[i] = sorted_y[j]
                sorted_y[j] = temp
            }
        }
    }
    
    # Assign ranks
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            if (x[i] == sorted_x[j]) {
                rank_x[i] = j
                break
            }
        }
        for (j = 1; j <= n; j++) {
            if (y[i] == sorted_y[j]) {
                rank_y[i] = j
                break
            }
        }
    }
    
    # Calculate Spearman correlation
    sum_d2 = 0
    for (i = 1; i <= n; i++) {
        d = rank_x[i] - rank_y[i]
        sum_d2 += d * d
    }
    
    if (n > 1) {
        return 1 - (6 * sum_d2) / (n * (n * n - 1))
    } else {
        return 0
    }
}

function pearson(x, y, n) {
    # Calculate means
    mean_x = 0
    mean_y = 0
    for (i = 1; i <= n; i++) {
        mean_x += x[i]
        mean_y += y[i]
    }
    mean_x /= n
    mean_y /= n
    
    # Calculate Pearson correlation
    numerator = 0
    sum_sq_x = 0
    sum_sq_y = 0
    for (i = 1; i <= n; i++) {
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        numerator += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    }
    
    if (sum_sq_x > 0 && sum_sq_y > 0) {
        return numerator / sqrt(sum_sq_x * sum_sq_y)
    } else {
        return 0
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
