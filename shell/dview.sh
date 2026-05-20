#!/usr/bin/env zsh


# Create a unique temporary file with a .smi or .sdf extension
MYTMP=$(mktemp -d)

trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

tmpfile=$MYTMP/smi.tsv
# Read stdin, convert spaces to tabs, write the header/data, and append properties
(
  cat | tr " " "\t" \
    | awk '
      BEGIN { FS="\t"; OFS="\t" }
      {
        if (NR==1){
          for (i=1;i<=NF;i++){
            printf "c%d%s", i, (i<NF ? "\t" : "\n")
          }
        }
        print $0
      }'
  cat <<'EOF'
<datawarrior properties>
<mainView="Table">
<mainViewCount="1">
<mainViewDockInfo0="root">
<mainViewName0="Table">
<mainViewType0="tableView">
</datawarrior properties>
EOF
) > "$tmpfile"

echo "$tmpfile"
# Open DataWarrior with the file in the background, then clean up the file
datawarrior "$tmpfile"
