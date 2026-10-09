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
    -p, --preview <id> Show transcript preview for a conversation ID
    -l, --list         List conversations without opening fzf
    -n <N>             Limit list to N conversations (default: all)
    -h, --help         Show this help message

In fzf:
    Select a conversation and press Enter to resume it with 'agy --conversation <id>'.
    Press Tab to toggle the preview window.
    Press Shift-Up/Down or Ctrl-U/Ctrl-D to scroll the preview window.
    Press Esc or Ctrl-C to cancel.
EOF
    exit 0
}

preview_conversation() {
    local conv_id="$1"
    python3 - "$conv_id" <<'PY'
import os
import json
import re
import sys
import shutil
import subprocess

conv_arg = sys.argv[1] if len(sys.argv) > 1 else ""
if not conv_arg:
    sys.exit(0)

# Extract uuid if a full line was passed (e.g. "2026-10-09 13:38  <uuid>  <prompt>")
uuid_match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", conv_arg, re.IGNORECASE)
conv_id = uuid_match.group(0) if uuid_match else conv_arg.strip()

brain_dir = os.path.expanduser("~/.gemini/antigravity-cli/brain")
log_file = os.path.join(brain_dir, conv_id, ".system_generated", "logs", "transcript.jsonl")

if not os.path.isfile(log_file):
    print(f"No transcript found for conversation {conv_id}")
    sys.exit(0)

BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
BLUE = "\033[1;38;5;75m"
GREEN = "\033[1;38;5;78m"
YELLOW = "\033[1;38;5;221m"
CYAN = "\033[38;5;80m"
GRAY = "\033[38;5;242m"
CODE_COLOR = "\033[38;5;222m"
RESET = "\033[0m"

bat_cmd = shutil.which("bat") or shutil.which("batcat")

def render_md(text):
    if bat_cmd:
        try:
            res = subprocess.run(
                [bat_cmd, "-l", "md", "--color=always", "--style=plain", "--paging=never", "--theme=ansi"],
                input=text,
                text=True,
                capture_output=True,
                timeout=3,
            )
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass

    # Built-in fallback renderer
    lines = text.splitlines()
    in_code_block = False
    rendered = []

    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            lang = line[3:].strip()
            if in_code_block:
                rendered.append(f"{GRAY}┌─── {lang or 'code'} ───{RESET}")
            else:
                rendered.append(f"{GRAY}└───{RESET}")
            continue

        if in_code_block:
            rendered.append(f"{GRAY}│{RESET} {CODE_COLOR}{line}{RESET}")
            continue

        # Headers
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            h_text = m.group(2)
            rendered.append(f"{YELLOW}{BOLD}{'#' * level} {h_text}{RESET}")
            continue

        # Blockquotes
        if line.startswith("> "):
            rendered.append(f"{GRAY}│{RESET} {ITALIC}{line[2:]}{RESET}")
            continue

        # Bullet points and numbers
        line = re.sub(r"^(\s*)[-*]\s+", rf"\1{CYAN}•{RESET} ", line)
        line = re.sub(r"^(\s*\d+\.)\s+", rf"{CYAN}\1{RESET} ", line)

        # Bold & Italic
        line = re.sub(r"\*\*\*(.*?)\*\*\*", rf"{BOLD}{ITALIC}\1{RESET}", line)
        line = re.sub(r"\*\*(.*?)\*\*", rf"{BOLD}\1{RESET}", line)
        line = re.sub(r"\*(.*?)\*", rf"{ITALIC}\1{RESET}", line)

        # Inline code
        line = re.sub(r"`([^`]+)`", rf"{CODE_COLOR}`\1`{RESET}", line)

        # Markdown links: [text](url) -> text (url)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", rf"{CYAN}\1{RESET} {GRAY}(\2){RESET}", line)

        rendered.append(line)

    return "\n".join(rendered)

print(f"{YELLOW}══════════════════════════════════════════════════════════════{RESET}")
print(f"{YELLOW} Conversation: {BOLD}{conv_id}{RESET}")
print(f"{YELLOW}══════════════════════════════════════════════════════════════{RESET}\n")

try:
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue

            msg_type = data.get("type")
            content = data.get("content") or ""

            if msg_type == "USER_INPUT" and content:
                # Strip user request wrappers and internal metadata tags
                clean = re.sub(r"<USER_REQUEST>\s*", "", content)
                clean = re.sub(r"</USER_REQUEST>.*", "", clean, flags=re.DOTALL)
                clean = re.sub(r"<[^>]+>", "", clean).strip()
                if clean:
                    print(f"{BLUE}▶ User:{RESET}")
                    print(render_md(clean))
                    print()
            elif msg_type == "PLANNER_RESPONSE" and content.strip():
                clean = content.strip()
                print(f"{GREEN}▶ Antigravity:{RESET}")
                print(render_md(clean))
                print(f"{GRAY}{'─' * 50}{RESET}\n")
except Exception as e:
    print(f"Error reading transcript: {e}")
PY
}

LIST_ONLY=0
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--preview)
            preview_conversation "$2"
            exit 0
            ;;
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

            prompts = []
            try:
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if '"USER_INPUT"' not in line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("type") == "USER_INPUT" and data.get("content"):
                                clean = re.sub(r"<USER_REQUEST>\s*", "", data["content"])
                                clean = re.sub(r"</USER_REQUEST>.*", "", clean, flags=re.DOTALL)
                                clean = re.sub(r"<[^>]+>", "", clean).strip()
                                if clean:
                                    prompts.append(clean.replace("\t", " ").replace("\n", " "))
                        except Exception:
                            continue
            except Exception:
                pass

            first_prompt = prompts[0][:100] if prompts else ""
            all_text = " ".join(prompts) if prompts else ""
            convs.append((mtime, entry.name, first_prompt, all_text))

convs.sort(key=lambda x: x[0], reverse=True)

if limit is not None:
    convs = convs[:limit]

for mtime, cid, first_p, all_p in convs:
    dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    print(f"{dt}  {cid}  {first_p}\t{all_p}")
PY
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
    generate_list | cut -f1
    exit 0
fi

if ! command -v fzf >/dev/null 2>&1; then
    echo "Warning: fzf is not installed, falling back to listing mode." >&2
    generate_list | cut -f1
    exit 0
fi

SCRIPT_PATH=$(readlink -f "$0")

SELECTED=$(generate_list | fzf \
    --ansi \
    --delimiter=$'\t' \
    --with-nth=1 \
    --prompt="Select agy conversation > " \
    --header="ENTER: resume | TAB: toggle | Shift-Up/Down or Ctrl-U/D: scroll | ESC: quit" \
    --reverse \
    --no-mouse \
    --preview="\"$SCRIPT_PATH\" --preview {1}" \
    --preview-window="right:60%:wrap" \
    --bind="tab:toggle-preview" \
    --bind="shift-up:preview-up,shift-down:preview-down" \
    --bind="ctrl-u:preview-page-up,ctrl-d:preview-page-down")

if [[ -n "$SELECTED" ]]; then
    CONV_ID=$(echo "$SELECTED" | cut -f1 | awk '{print $3}')
    if [[ -n "$CONV_ID" ]]; then
        echo "Resuming conversation: $CONV_ID"
        exec agy --conversation "$CONV_ID"
    fi
fi
