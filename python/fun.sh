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
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) | ./project.py --method tsne
}

test_pca () {
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) | ./project.py --method pca
}

test_tsne_labels () {
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100) | ./project.py --method tsne -l
}

test_tsne_labels_text () {
    paste =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100 | shuf) =(seq 100) \
        | awk '{if (NR<50){print $0,"a"}else{print $0,"b"}}' |./project.py --method tsne -l -t
}

mk_test.rec () {
    OUTFILE="data/test.rec"
    for i in $(seq 10); do
        echo "index=$i"
        sq=$(calc -t "$i**2")
        echo "square=$sq"
        echo "--"
    done >! $OUTFILE
    echo "$OUTFILE created"
}

test_rec () {
    set -x
    cat data/test.rec | rec -f index square
}
