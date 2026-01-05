#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue May 13 09:28:31 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

REMOTEHOST="maestro"
REMOTEDIR="/pasteur/gaia/homes/bougui/archives"

ssh "$REMOTEHOST" "mkdir -p $REMOTEDIR"

# Function to clean up on exit
cleanup() {
    # Remove any incomplete archives that might have been created
    if [[ -n "$INCOMPLETE_ARCHIVE" ]]; then
        ssh "$REMOTEHOST" "rm -f $INCOMPLETE_ARCHIVE" 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

function usage () {
    cat << EOF
Usage: $(basename "$0") <file>
Archive files given in the file <file>.
The file <file> should contain a list of files to be archived, one per line.
The files will be archived on the remote host $REMOTEHOST in the directory $REMOTEDIR.
The original files will be removed and replaced by a script that will unarchive them.
The script will be called <file>.arc.sh and will be created in the same directory as the original file.

    -h, --help print this help message and exit
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 0
fi

if [[ -f $1 ]]; then
    # the given file contains a list of files to be archived
    NLINES=$(wc -l < "$1")
    echo $NLINES files to be archived
    i=1
    T0=$SECONDS  # SECONDS: built-in variable that returns the number of seconds since the script started
    for FILE in $(cat "$1"); do
        # print a progress bar
        PROGRESS=$(echo "scale=2; $i*100/$NLINES" | bc)
        ELAPSED=$((SECONDS - T0))
        REMAINING=$((ELAPSED * (NLINES - i) / i))
        ETA=$(printf "%02d:%02d:%02d" $((REMAINING / 3600)) $(((REMAINING % 3600) / 60)) $((REMAINING % 60)))
        printf "\rArchiving files: [%-50s] %.2f%% ETA: %s " $(printf '#%.0s' $(seq 1 $((i*50/NLINES)))) $PROGRESS $ETA
        i=$((i+1))
        if [ -d "$FILE" ]; then
            # It's a directory, archive it using tar
            DIRNAME=$(dirname $(realpath "$FILE"))
            ssh "$REMOTEHOST" "mkdir -vp ${REMOTEDIR}${DIRNAME}"
            OUTFILE="${REMOTEDIR}$(realpath "$FILE").tar.gz"
            
            # Set the incomplete archive variable for cleanup
            INCOMPLETE_ARCHIVE="$OUTFILE"
            
            # Store the original directory's timestamp
            ORIGINAL_TIMESTAMP=$(stat -c %Y:%y "$FILE")
            
            # Create a tar archive, compress it with pigz, and send it to the remote host
            tar -czf - -C "$(dirname $(realpath "$FILE"))" "$(basename $(realpath "$FILE"))" | ssh "$REMOTEHOST" "cat > $OUTFILE"
            
            # check if the file was successfully archived
            if ssh "$REMOTEHOST" "test -f $OUTFILE"; then
                echo "Directory $FILE archived as $OUTFILE" > /dev/null
                # Clear the incomplete archive variable since it's now complete
                INCOMPLETE_ARCHIVE=""
            else
                echo "Error archiving directory $FILE" > /dev/stderr
                exit 1
            fi
            
            # Create the unarchive script
            cat << EOF > "${FILE}.arc.sh"
#!/usr/bin/env bash
# This script will unarchive the directory $REMOTEHOST:$OUTFILE
# It was created by the script $(basename "$0")
# Do not edit it
# Transfer the compressed file and decompress locally
ssh "$REMOTEHOST" "cat $OUTFILE" | tar -xzf - -C "$(dirname $(realpath "$FILE"))"
# Restore the original timestamp
ORIGINAL_TIMESTAMP="${ORIGINAL_TIMESTAMP}"
if [[ -n "\$ORIGINAL_TIMESTAMP" ]]; then
    IFS=':' read -r mtime atime <<< "\$ORIGINAL_TIMESTAMP"
    touch -d "@\$mtime" "$(dirname $(realpath "$FILE"))/$(basename $(realpath "$FILE"))"
fi
# check if the directory was successfully unarchived
if [[ -d "$(dirname $(realpath "$FILE"))/$(basename $(realpath "$FILE"))" ]]; then
    echo "Directory $(dirname $(realpath "$FILE"))/$(basename $(realpath "$FILE")) unarchived"
else
    echo "Error unarchiving directory $(dirname $(realpath "$FILE"))/$(basename $(realpath "$FILE"))"
    exit 1
fi
# remove the archived file on the remote host
ssh "$REMOTEHOST" "rm -v $OUTFILE"
# remove the script itself
rm -v "$(realpath "$FILE").arc.sh"
EOF
            chmod +x "${FILE}.arc.sh"
            # AI! give the ORIGINAL_TIMESTAMP to the file: "${FILE}.arc.sh"
            # remove the original directory
            rm -rv "$FILE"
        elif [ -f "$FILE" ] && [ ! -L "$FILE" ]; then
            # It's a regular file, archive it as before
            DIRNAME=$(dirname $(realpath "$FILE"))
            ssh "$REMOTEHOST" "mkdir -vp ${REMOTEDIR}${DIRNAME}"
            OUTFILE="${REMOTEDIR}$(realpath $FILE).gz"
            
            # Set the incomplete archive variable for cleanup
            INCOMPLETE_ARCHIVE="$OUTFILE"
            
            # Store the original file's timestamp
            ORIGINAL_TIMESTAMP=$(stat -c %Y:%y "$FILE")
            
            # cat the file to gzip and send it to the remote host
            pcat "$FILE" | pigz -c | ssh "$REMOTEHOST" "cat > $OUTFILE"
            
            # check if the file was successfully archived
            if ssh "$REMOTEHOST" "test -f $OUTFILE"; then
                echo "File $FILE archived as $OUTFILE" > /dev/null
                # Clear the incomplete archive variable since it's now complete
                INCOMPLETE_ARCHIVE=""
            else
                echo "Error archiving file $FILE" > /dev/stderr
                exit 1
            fi
            
            # Create the unarchive script
            cat << EOF > "${FILE}.arc.sh"
#!/usr/bin/env bash
# This script will unarchive the file $REMOTEHOST:$OUTFILE
# It was created by the script $(basename "$0")
# Do not edit it
# Transfer the compressed file and decompress locally
ssh "$REMOTEHOST" "cat $OUTFILE" | pigz -c -d > "$(realpath "$FILE")"
# Restore the original timestamp
ORIGINAL_TIMESTAMP="${ORIGINAL_TIMESTAMP}"
if [[ -n "\$ORIGINAL_TIMESTAMP" ]]; then
    IFS=':' read -r mtime atime <<< "\$ORIGINAL_TIMESTAMP"
    touch -d "@\$mtime" "$(realpath "$FILE")"
fi
# check if the file was successfully unarchived
if [[ -f "$(realpath "$FILE")" ]]; then
    echo "File $(realpath "$FILE") unarchived"
else
    echo "Error unarchiving file $(realpath "$FILE")"
    exit 1
fi
# remove the archived file on the remote host
ssh "$REMOTEHOST" "rm -v $OUTFILE"
# remove the script itself
rm -v "$(realpath "$FILE").arc.sh"
EOF
            chmod +x "${FILE}.arc.sh"
            # remove the original file
            rm "$FILE"
        else
            if [ -f ${FILE}.arc.sh ]; then
                echo "File $FILE is already archived"
                NLINES=$((NLINES-1))
                i=$((i-1))
                T0=$SECONDS
            else
                echo "File $FILE is not a regular file or directory"
            fi
        fi
    done
else
    echo "File $1 does not exist or is not a regular file"
    exit 1
fi
