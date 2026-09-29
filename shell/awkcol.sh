#!/usr/bin/env bash
# awkcol.sh - Helper CLI tool to find column indexes for awk
# Works with any text stream, log line, table output, or delimited file (CSV/TSV/etc.)

set -euo pipefail

usage() {
    cat << 'EOF'
awkcol.sh - Find awk column indices for any input or file

Usage:
  awkcol.sh [options] [file]
  <command> | awkcol.sh [options]

Options:
  -F <delim>     Set field delimiter (default: awk whitespace rule)
  -n <line_num>  Inspect a specific line number (default: 1)
  -s <pattern>   Filter/search columns matching a regex or string
  -h, --help     Show this help message

Examples:
  # Inspect whitespace-separated command output:
  ls -l | awkcol.sh -n 2
  ps aux | awkcol.sh -s "nginx"

  # Delimited files:
  awkcol.sh -F: -n 1 /etc/passwd
  awkcol.sh -F, -n 5 data.csv
EOF
    exit 0
}

DELIM=""
LINE_NUM=1
SEARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -F)
            if [[ $# -lt 2 ]]; then
                echo "Error: -F requires a delimiter argument." >&2
                exit 1
            fi
            DELIM="$2"
            shift 2
            ;;
        -n)
            if [[ $# -lt 2 ]]; then
                echo "Error: -n requires a line number." >&2
                exit 1
            fi
            LINE_NUM="$2"
            shift 2
            ;;
        -s)
            if [[ $# -lt 2 ]]; then
                echo "Error: -s requires a search pattern." >&2
                exit 1
            fi
            SEARCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            break
            ;;
    esac
done

awk -v delim="$DELIM" -v target_line="$LINE_NUM" -v search="$SEARCH" '
BEGIN {
    if (delim != "") {
        FS = delim
    }
}
NR == target_line {
    found = 0
    for (i = 1; i <= NF; i++) {
        if (search == "" || $i ~ search) {
            printf "$%-3d : %s\n", i, $i
            found = 1
        }
    }
    if (search != "" && !found) {
        print "No matching columns found for pattern: " search > "/dev/stderr"
    }
    exit
}
' "${1:-/dev/stdin}"
