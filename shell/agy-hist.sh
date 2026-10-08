#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Oct  8 16:23:00 2026

set -e
set -o pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Interactive fuzzy navigator for Antigravity (agy) conversation history.

Options:
    -l, --list        List conversations without opening fzf
    -n <N>            Limit list to N conversations (default: all)
    -h, --help        Show this help message

In fzf:
    Select a conversation and press Enter to resume it with 'agy --conversation <id>'.
    Press Esc or Ctrl-C to cancel.
EOF
    exit 0
}

LIST_ONLY=0
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--list)
            LIST_ONLY=1
            shift
            ;;
        -n)
            LIMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

export AGY_LIMIT="$LIMIT"

# Generate formatted conversation list using python
generate_list() {
    python3 - <<'PY'
import os
import json
import datetime
import re
import sys

brain_dir = os.path.expanduser("~/.gemini/antigravity-cli/brain")
limit_str = os.environ.get("AGY_LIMIT", "").strip()
limit = int(limit_str) if limit_str.isdigit() else None

if not os.path.isdir(brain_dir):
    sys.exit(0)

convs = []

for entry in os.scandir(brain_dir):
    if entry.is_dir():
        log_file = os.path.join(entry.path, ".system_generated", "logs", "transcript.jsonl")
        if os.path.isfile(log_file):
            try:
                mtime = os.path.getmtime(log_file)
            except OSError:
                continue

            first_prompt = ""
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "USER_INPUT" and data.get("content"):
                                clean = re.sub(r"<[^>]+>", "", data["content"])
                                first_prompt = clean.strip().replace("\n", " ")[:100]
                                break
                        except Exception:
                            continue
            except Exception:
                pass

            convs.append((mtime, entry.name, first_prompt))

convs.sort(key=lambda x: x[0], reverse=True)

if limit is not None:
    convs = convs[:limit]

for mtime, cid, prompt in convs:
    dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    print(f"{dt}  {cid}  {prompt}")
PY
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    generate_list
    exit 0
fi

if ! command -v fzf >/dev/null 2>&1; then
    echo "Warning: fzf is not installed, falling back to listing mode." >&2
    generate_list
    exit 0
fi

SELECTED=$(generate_list | fzf \
    --prompt="Select agy conversation > " \
    --header="ENTER: resume conversation | ESC: quit" \
    --reverse \
    --no-mouse)

if [[ -n "$SELECTED" ]]; then
    CONV_ID=$(echo "$SELECTED" | awk '{print $3}')
    if [[ -n "$CONV_ID" ]]; then
        echo "Resuming conversation: $CONV_ID"
        exec agy --conversation "$CONV_ID"
    fi
fi
