#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-09-15
# Description: Maestro cluster group 'bis' FairShare & resource usage report

set -e
set -o pipefail

# ANSI color codes when connected to a terminal
if [ -t 1 ]; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_DIM="\033[2m"
    C_CYAN="\033[36m"
    C_GREEN="\033[32m"
    C_YELLOW="\033[33m"
    C_RED="\033[31m"
    C_BLUE="\033[34m"
else
    C_RESET="" C_BOLD="" C_DIM="" C_CYAN="" C_GREEN="" C_YELLOW="" C_RED="" C_BLUE=""
fi

SEP="================================================================================"
SUBSEP="--------------------------------------------------------------------------------"

echo -e "${C_BOLD}${C_CYAN}${SEP}${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}         MAESTRO CLUSTER — ACCOUNT BIS FAIRSHARE & USAGE DASHBOARD${C_RESET}"
echo -e "${C_DIM}$(date '+%Y-%m-%d %H:%M:%S')${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}${SEP}${C_RESET}\n"

# 1. Fetch data from Maestro over a single SSH session
raw_dump=$(ssh -q maestro 'bash -s' << 'REMOTE'
scontrol show config 2>/dev/null | awk -F'= ' '/PriorityDecayHalfLife/{print $2}'
echo "===SSHARE==="
sshare -A bis -a -P -o Account,User,NormShares,RawUsage,NormUsage,EffectvUsage,FairShare 2>/dev/null
echo "===SQUEUE==="
squeue -A bis -h -o "%u|%P|%t|%r" 2>/dev/null
REMOTE
)

halflife=$(echo "$raw_dump" | sed -n '1p')
sshare_data=$(echo "$raw_dump" | sed -n '/===SSHARE===/,/===SQUEUE===/{ /===SSHARE===/d; /===SQUEUE===/d; p }')
squeue_data=$(echo "$raw_dump" | sed -n '/===SQUEUE===/,$ { /===SQUEUE===/d; p }')

# 2. Section 1: Overview
echo -e "${C_BOLD}${C_BLUE}[1] GROUP FAIRSHARE & CONSUMPTION OVERVIEW${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"
echo -e "  • Account Name                    : ${C_BOLD}bis${C_RESET}"
echo -e "  • Priority Decay Half-Life        : ${C_BOLD}${halflife:-7 days}${C_RESET} (rolling exponential decay)"

echo "$sshare_data" | awk -F'|' -v C_BOLD="$C_BOLD" -v C_RED="$C_RED" -v C_RESET="$C_RESET" '
NR==2 {
    ratio = ($3 > 0 ? $5 / $3 : 0);
    printf "  • Allocated Cluster Share (Target): %s%.2f%%%s\n", C_BOLD, $3*100, C_RESET;
    printf "  • Recent Cluster Consumption     : %s%s%.2f%%%s (%s%.1fx target%s)\n", C_BOLD, C_RED, $5*100, C_RESET, C_RED, ratio, C_RESET;
    printf "  • Total Raw Usage (decayed pts)   : %'"'"'d\n", $4;
}
'
echo ""

# 3. Section 2: Per-user table
echo -e "${C_BOLD}${C_BLUE}[2] PER-USER CONSUMPTION (Sorted by recent usage)${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

(
  echo -e "USER|SHARE_%|RAW_USAGE|CLUSTER_%|GROUP_%|FAIRSHARE"
  echo "$sshare_data" | awk -F'|' '
  NR==2 {
      group_raw = $4 + 0
  }
  NR>2 && $2!="" && $4>0 {
      group_pct = (group_raw > 0 ? ($4 / group_raw) * 100 : 0)
      printf "%s|%.2f%%|%d|%.2f%%|%.2f%%|%s\n", $2, $3*100, $4, $5*100, group_pct, $7
  }' | sort -t'|' -k3,3nr | awk -F'|' '{ printf "%s|%s|%'"'"'d|%s|%s|%s\n", $1, $2, $3, $4, $5, $6 }'
) | column -t -s '|'

inactive=$(echo "$sshare_data" | awk -F'|' 'NR>2 && $2!="" && ($4=="" || $4==0) { printf "%s ", $2 }')
if [ -n "$inactive" ]; then
    echo -e "\n${C_DIM}• Inactive users (0 recent usage, FairShare ~0.048): $inactive${C_RESET}"
fi
echo ""

# 4. Section 3: Live Jobs
echo -e "${C_BOLD}${C_BLUE}[3] LIVE JOBS IN GROUP BIS${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

if [ -z "$squeue_data" ]; then
    echo -e "  ${C_GREEN}No active or pending jobs currently queued for account bis.${C_RESET}"
else
    (
      echo -e "USER|PARTITION|STATE|COUNT|REASON"
      echo "$squeue_data" | sort | uniq -c | awk '{ printf "%s|%s|%s|%s|%s\n", $2, $3, $4, $1, $5 }'
    ) | column -t -s '|'
fi
echo ""

# 5. Section 4: Summary & Diagnostic
echo -e "${C_BOLD}${C_BLUE}[4] DIAGNOSTIC & PRIORITY SUMMARY${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

echo "$sshare_data" | awk -F'|' -v C_BOLD="$C_BOLD" -v C_RED="$C_RED" -v C_GREEN="$C_GREEN" -v C_RESET="$C_RESET" -v curr_user="${USER:-bougui}" '
NR==2 {
    group_target = $3 + 0
    group_raw = $4 + 0
    group_cluster = $5 + 0
    ratio = (group_target > 0 ? group_cluster / group_target : 0)
}
NR>2 && $2!="" && $4>0 {
    raw = $4 + 0
    cluster_pct = $5 * 100
    group_pct = (group_raw > 0 ? (raw / group_raw) * 100 : 0)
    
    if (raw > top_raw) {
        top_raw = raw
        top_user = $2
        top_group_pct = group_pct
        top_cluster_pct = cluster_pct
    }
    if ($2 == curr_user || $2 == "bougui") {
        user_found = 1
        user_name = $2
        user_group_pct = group_pct
        user_cluster_pct = cluster_pct
    }
}
END {
    if (top_user != "") {
        printf "  1. %sMain Consumer:%s %s%s%s accounts for %s~%.1f%%%s of group bis usage (~%.2f%% of entire cluster).\n", C_BOLD, C_RESET, C_RED, top_user, C_RESET, C_BOLD, top_group_pct, C_RESET, top_cluster_pct;
    } else {
        printf "  1. %sMain Consumer:%s None (no active usage)\n", C_BOLD, C_RESET;
    }
    if (user_found) {
        printf "  2. %sYour Usage (%s):%s %s~%.1f%%%s of group usage (~%.2f%% of cluster).\n", C_BOLD, user_name, C_RESET, C_GREEN, user_group_pct, C_RESET, user_cluster_pct;
    } else {
        printf "  2. %sYour Usage (%s):%s %s0.0%%%s of group usage (0.00%% of cluster).\n", C_BOLD, curr_user, C_RESET, C_GREEN, C_RESET;
    }
    printf "  3. %sFairShare Impact:%s SLURM evaluates FairShare globally across CPU & GPU. Because group bis\n", C_BOLD, C_RESET;
    if (ratio > 1) {
        printf "     overall usage exceeds target allocation by ~%.1fx, all pending jobs receive reduced FairShare priority.\n", ratio;
    } else {
        printf "     overall usage is within target allocation (~%.2fx target), FairShare priority remains healthy.\n", ratio;
    }
}
'
echo -e "${C_BOLD}${C_CYAN}${SEP}${C_RESET}\n"
