#!/usr/bin/env zsh


# Create a unique temporary file with a .tsv extension
MYTMP=$(mktemp -d)

trap 'rm -rf "$MYTMP"' EXIT INT TERM  # Clean up if script exits/fails before launching

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
(
  datawarrior "$tmpfile" < /dev/null
  rm -rf "$MYTMP"
) >/dev/null 2>&1 &

# Clear the trap so the temporary directory is not deleted immediately upon script exit
trap - EXIT INT TERM




