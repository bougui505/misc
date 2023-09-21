#!/usr/bin/env zsh
cat << EOF

$(date): sourcing $0
EOF

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap '/bin/rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

precmd() {
    [ -f fun.sh ] && source fun.sh
}

test_tsne () {
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) | ./tsne.py
}

test_tsne_labels () {
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100) | ./tsne.py -l
}
