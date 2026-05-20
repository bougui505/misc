#!/usr/bin/env zsh


# Create a unique temporary file with a .smi or .sdf extension
MYTMP=$(mktemp -d)

trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

tmpfile=$MYTMP/smi.csv
# Read stdin and dump it into the temp file
cat | tr " " "," \
  | awkcsv '
    {
      if (NR==1){
        for (i=1;i<=NF;i++){
          printf "c%d%s", i, (i<NF ? "," : "\n")
        }
      }
      print $0
    }' > $tmpfile
echo $tmpfile
# Open DataWarrior with the file in the background, then clean up the file
datawarrior "$tmpfile"
