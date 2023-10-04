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
    for x in $(seq 10); do
        echo "x=$x"
        if [ $x -eq 6 ]; then
            echo "--"
            continue
        fi
        x2=$(calc -t "$x**2")
        echo "x2=$x2"
        x3=$(calc -t "$x**3")
        echo "x3=$x3"
        echo "--"
    done >! $OUTFILE
    echo "$OUTFILE created"

    OUTFILE="data/test2.rec"
    for x in $(seq 3 8 | shuf); do
        echo "x=$x"
        x3=$(calc -t "$x**3")
        echo "x3=$x3"
        x4=$(calc -t "$x**4")
        echo "x4=$x4"
        echo "--"
    done >! $OUTFILE
    echo "$OUTFILE created"
}

test_rec () {
    # mk_test.rec
    set -x
    cat data/test.rec | rec -f x x2
    rec -f x x2 --file data/test.rec
    cat data/test.rec | rec
    rec --merge data/test.rec data/test2.rec
    cat data/test.rec | rec --sel "x2==64 or x==2" -r

}
