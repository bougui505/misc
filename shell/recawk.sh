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

function usage () {
    cat << EOF
Usage: recawk [OPTIONS] 'AWK_SCRIPT' [FILES]

A powerful tool to process record-formatted files (key=value) using AWK.

Options:
  -h, --help           Print this help message and exit
  -n, --nrec           Print the number of records
  -s, --sample N       Pick N random records (reservoir sampling)
  -k, --keys           Print all unique keys present in the file
  --torec SEP          Convert column-based files (e.g., CSV/TSV) to rec format
  --tocsv              Convert rec format to CSV (first record defines columns)
  -v VAR=VAL           Pass a variable to the AWK script

Record Format:
  Records are separated by '--' on a line by itself.
  Example:
    key1=val1
    key2=val2
    --
    key1=val3
    key2=val4
    --

AWK Integration:
  - Each record is loaded into the 'rec' associative array.
  - Access fields using: rec["key1"]
  - Current record count is available in 'nr' and 'fnr'.
  - Predefined functions:
      printrec()       - Print the current record in key=val format
      spearman(x,y,n)  - Compute Spearman correlation for arrays of length n
      pearson(x,y,n)   - Compute Pearson correlation for arrays of length n

Performance Optimizations:
  - Smart Filtering: recawk detects used fields (e.g., rec["tmscore"]) and pre-filters
    the input using 'grep' to significantly speed up processing of large files.
  - Mawk Support: Automatically uses 'mawk' instead of 'gawk' for a ~20% speedup, 
    unless gawk-specific features (like --torec) are used.
  - Pair with pigz: For maximum speed on .gz files, use: pigz -dc file.rec.gz | recawk ...

Examples:
  # Extract a single field from a compressed file
  zcat data.rec.gz | recawk '{print rec["tmscore"]}'

  # Compute correlation between two fields
  cat data.rec | recawk '{x[nr]=rec["x"]; y[nr]=rec["y"]} END {print spearman(x,y,nr)}'

  # Sample 100 random records
  recawk --sample 100 data.rec

  # Convert CSV to rec format
  cat data.csv | recawk --torec ","
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

CMD=$(echo "$1" | tr "\n" "$" | gawk -F"END" '{print $1}' | tr "$" "\n")
ENDCMD=$(echo "$1" | tr "\n" "$" | gawk -F"END" '{print $2}' | tr "$" "\n")
FILENAMES="${@:2}"

# Smart field filtering
FILTER=""
if [[ $TOREC == 0 && $KEYS == 0 && $TOCSV == 0 ]]; then
    # Detect fields used in the script: rec["field"] or rec['field']
    USED_FIELDS=$(echo "$1" | grep -oP "rec\[\s*['\"]\K[^'\"]+(?=['\"]\s*\])" | sort -u)
    # If the script iterates over all fields or uses printrec, we can't filter
    if [[ -n $USED_FIELDS ]] && ! echo "$1" | grep -qE "printrec|field\s+in\s+rec"; then
        FILTER="^--$"
        for f in $USED_FIELDS; do
            FILTER="$FILTER|^$f="
        done
    fi
fi

AWK_BIN="gawk"
if command -v mawk > /dev/null 2>&1 && [[ $TOREC == 0 ]]; then
    AWK_BIN="mawk"
fi

if [[ $GETNREC -eq 1 ]]; then
    getnrec $FILENAMES
    exit 0
fi

if [[ $KEYS -eq 1 ]]; then
    gawk -F"=" '{
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
    gawk '
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
        gawk -v FPAT="[^[:space:]]+|(\"([^\"]|\"\")*\")" '{
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
        gawk -v FPAT="[^$TOREC]*|(\"([^\"]|\"\")*\")" '{
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
  # We no longer need to cat to a temp file or run getnrec!
    # We change the CMD to store the record in a reservoir instead of printing immediately.
    V="SAMPLE=$SAMPLE"
    # We wrap the user's command to run only at the END on the sampled records
    CMD='{
        # Reservoir Sampling Logic
        if (nr < SAMPLE) {
            # Fill the reservoir initially
            for (key in rec) reservoir[nr, key] = rec[key]
            res_keys[nr] = 1
        } else {
            # Replace with decreasing probability
            r = int((nr + 1) * rand())
            if (r < SAMPLE) {
                # Clear old record at index r and replace
                for (key in rec) {
                    # We use a 2D array simulation to store multiple records
                    reservoir[r, key] = rec[key]
                }
            }
        }
    }'
    # At the END, we loop through the reservoir and run the user's code
    ENDCMD='
        for (i=0; i < SAMPLE; i++) {
            # Restore the "rec" array for the current sampled record
            delete rec
            for (combined_key in reservoir) {
                split(combined_key, parts, SUBSEP)
                if (parts[1] == i) {
                    rec[parts[2]] = reservoir[combined_key]
                }
            }
            printrec();
            print("--")
            # Simulate the NR/FNR for the sample and run user command
            nr = i + 1; fnr = i + 1;
        }
    '
fi

# Define the AWK script parts to avoid duplication
AWK_FUNCTIONS=$(cat << 'EOF'
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
EOF
)

AWK_MAIN_LOOP_BEGIN=$(cat << 'EOF'
BEGIN{
srand(seed)
nr=0
}
{
if (FNR==1){
    fnr=0
}
if ($0=="--"){
    nr+=1
    fnr+=1
EOF
)

AWK_MAIN_LOOP_END=$(cat << 'EOF'
    delete rec
}
else{
    rec[$1]=substr($0,length($1)+2)
}
}
END{
EOF
)

FULL_AWK_SCRIPT="${AWK_FUNCTIONS}
${AWK_MAIN_LOOP_BEGIN}
${CMD}
${AWK_MAIN_LOOP_END}
${ENDCMD}
}"

if [[ -n $FILTER ]]; then
    if [[ -z $FILENAMES ]]; then
        grep -E "$FILTER" | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
    else
        grep -E "$FILTER" $FILENAMES | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
    fi
else
    $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT" $FILENAMES
fi
