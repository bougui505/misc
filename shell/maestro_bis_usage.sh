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
echo "===SSHARE_ALL==="
sshare -P -o Account,User,NormShares,RawUsage,NormUsage,EffectvUsage,FairShare 2>/dev/null
echo "===SSHARE_BIS==="
sshare -A bis -a -P -o Account,User,NormShares,RawUsage,NormUsage,EffectvUsage,FairShare 2>/dev/null
echo "===SQUEUE_BIS==="
squeue -A bis -h -o "%u|%P|%t|%r" 2>/dev/null
echo "===SQUEUE_ALL==="
squeue -p common,gpu,dedicatedgpu -h -o "%P|%t|%r" 2>/dev/null
REMOTE
)

halflife=$(echo "$raw_dump" | sed -n '1p')
sshare_all_data=$(echo "$raw_dump" | sed -n '/===SSHARE_ALL===/,/===SSHARE_BIS===/{ /===SSHARE_ALL===/d; /===SSHARE_BIS===/d; p }')
sshare_data=$(echo "$raw_dump" | sed -n '/===SSHARE_BIS===/,/===SQUEUE_BIS===/{ /===SSHARE_BIS===/d; /===SQUEUE_BIS===/d; p }')
squeue_data=$(echo "$raw_dump" | sed -n '/===SQUEUE_BIS===/,/===SQUEUE_ALL===/{ /===SQUEUE_BIS===/d; /===SQUEUE_ALL===/d; p }')
squeue_all_data=$(echo "$raw_dump" | sed -n '/===SQUEUE_ALL===/,$ { /===SQUEUE_ALL===/d; p }')

# 2. Section 1: Overview
echo -e "${C_BOLD}${C_BLUE}[1] GROUP BIS FAIRSHARE & CONSUMPTION OVERVIEW${C_RESET}"
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

# 3. Section 2: Top 10 groups table
echo -e "${C_BOLD}${C_BLUE}[2] TOP 10 GROUPS CLUSTER CONSUMPTION (Sorted by recent usage)${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

(
  echo -e "RANK|GROUP|USAGE_%|TARGET_%|VS_TARGET|RAW_USAGE"
  echo "$sshare_all_data" | awk -F'|' '
  NR==2 { total_raw = $4 + 0 }
  NR>2 && $1!="" && ($2 == "" || $2 ~ /^[ \t]*$/) {
      acc = $1; gsub(/^[ \t]+|[ \t]+$/, "", acc);
      if (acc == "root") next;
      raw = $4 + 0;
      target = $3 * 100;
      pct = (total_raw > 0 ? (raw / total_raw) * 100 : 0);
      ratio = (target > 0 ? (pct / target) : 0);
      if (raw > 0) {
          printf "%s|%.2f%%|%.2f%%|%.1fx|%d\n", acc, pct, target, ratio, raw;
      }
  }' | sort -t'|' -k5,5nr | awk -F'|' -v C_BOLD="$C_BOLD" -v C_CYAN="$C_CYAN" -v C_RESET="$C_RESET" '
  {
      rank++;
      acc = $1;
      if (acc == "bis") {
          bis_rank = rank;
          bis_line = sprintf("%d|%s%s%s (you)|%s|%s|%s|%'"'"'d", rank, C_BOLD C_CYAN, acc, C_RESET, $2, $3, $4, $5);
      }
      if (rank <= 10) {
          acc_display = (acc == "bis" ? sprintf("%s%s%s (you)", C_BOLD C_CYAN, acc, C_RESET) : acc);
          printf "%d|%s|%s|%s|%s|%'"'"'d\n", rank, acc_display, $2, $3, $4, $5;
      }
  }
  END {
      if (bis_rank > 10) {
          print "...|...|...|...|...|...";
          print bis_line;
      }
  }'
) | column -t -s '|'

active_groups=$(echo "$sshare_all_data" | awk -F'|' '
NR>2 && $1!="" && ($2 == "" || $2 ~ /^[ \t]*$/) {
    acc = $1; gsub(/^[ \t]+|[ \t]+$/, "", acc);
    if (acc == "root") next;
    if ($4 > 0) count++;
}
END { print count+0 }')

inactive_groups=$(echo "$sshare_all_data" | awk -F'|' '
NR>2 && $1!="" && ($2 == "" || $2 ~ /^[ \t]*$/) {
    acc = $1; gsub(/^[ \t]+|[ \t]+$/, "", acc);
    if (acc == "root") next;
    if ($4 == 0 || $4 == "") count++;
}
END { print count+0 }')
echo -e "\n${C_DIM}• Showing top 10 of ${active_groups} active groups (${inactive_groups} groups have 0% recent usage).${C_RESET}"
echo ""

# 4. Section 3: Per-user table for group bis
echo -e "${C_BOLD}${C_BLUE}[3] PER-USER CONSUMPTION IN GROUP BIS (Sorted by recent usage)${C_RESET}"
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

# 5. Section 4: Live Jobs in Group BIS
echo -e "${C_BOLD}${C_BLUE}[4] LIVE JOBS IN GROUP BIS${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

if [ -z "$squeue_data" ]; then
    echo -e "  ${C_GREEN}No active or pending jobs currently queued for account bis.${C_RESET}"
else
    (
      echo -e "USER|PARTITION|STATE|COUNT|REASON"
      echo "$squeue_data" | awk -F'|' '
      NF>=4 {
          user = $1; gsub(/^[ \t]+|[ \t]+$/, "", user);
          key = user "|" $2 "|" $3 "|" $4;
          count[key]++;
      }
      END {
          for (k in count) {
              split(k, p, "|");
              printf "%s|%s|%s|%d|%s\n", p[1], p[2], p[3], count[k], p[4];
          }
      }' | sort -t'|' -k1,1 -k2,2 -k3,3
    ) | column -t -s '|'
fi
echo ""

# 6. Section 5: Live Jobs in Common, GPU & DedicatedGPU Partitions (Global)
echo -e "${C_BOLD}${C_BLUE}[5] LIVE JOBS IN COMMON, GPU & DEDICATEDGPU PARTITIONS (Global)${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

if [ -z "$squeue_all_data" ]; then
    echo -e "  ${C_GREEN}No active or pending jobs currently queued in common, gpu, or dedicatedgpu partitions.${C_RESET}"
else
    (
      echo -e "PARTITION|STATE|COUNT|REASON"
      echo "$squeue_all_data" | awk -F'|' '
      NF>=3 && ($1 == "common" || $1 == "gpu" || $1 == "dedicatedgpu") {
          key = $1 "|" $2 "|" $3;
          count[key]++;
      }
      END {
          for (k in count) {
              split(k, p, "|");
              printf "%s|%s|%d|%s\n", p[1], p[2], count[k], p[3];
          }
      }' | sort -t'|' -k1,1 -k2,2
    ) | column -t -s '|'

    echo "$squeue_all_data" | awk -F'|' -v C_DIM="$C_DIM" -v C_RESET="$C_RESET" '
    NF>=2 && ($1 == "common" || $1 == "gpu" || $1 == "dedicatedgpu") {
        total++;
        if ($2 == "R") r++;
        else if ($2 == "PD") pd++;
        else other++;
    }
    END {
        printf "\n%s• Common, GPU & DedicatedGPU totals: %'"'"'d Running (R), %'"'"'d Pending (PD), %'"'"'d Other (%'"'"'d total jobs)%s\n", C_DIM, r, pd, other, total, C_RESET;
    }'
fi
echo ""

# 7. Section 6: Accessible Partitions for Account BIS
echo -e "${C_BOLD}${C_BLUE}[6] ACCESSIBLE PARTITIONS & CHARACTERISTICS FOR BIS${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

(
  echo -e "PARTITION|TYPE|NODES|CORES|HARDWARE / MEMORY|ALLOWED_QOS|MAX_WALLTIME|PREEMPT|TIER"
  echo -e "common|CPU Standard|41|3,936|37 TB RAM (~720 GB/node)|normal, fast, ultrafast|24h (normal) / 2h (fast)|No|1"
  echo -e "dedicated|CPU Opportunistic|112|10,752|85 TB RAM (up to 2 TB/node)|fast, ultrafast|2h (fast) / 5m (ultrafast)|Yes (Requeue)|5000"
  echo -e "gpu|GPU Standard|17|1,728|115 GPUs (A100, A40, L40S, RTX6000)|gpu, normal, fast, ultrafast|3d (gpu) / 24h (normal)|No|1000"
  echo -e "dedicatedgpu|GPU Opportunistic|9|528|45 GPUs (A100, A40, RTX6000)|fast, ultrafast|2h (fast) / 5m (ultrafast)|Yes (Requeue)|5000"
  echo -e "long|CPU Long Runs|4|384|2.8 TB RAM (~720 GB/node)|long|365d (long)|No|1"
  echo -e "clcgwb|CPU CLC Workbench|1|96|720 GB RAM|normal, fast, ultrafast|24h (normal) / 2h (fast)|No|10000"
) | column -t -s '|'

echo -e "\n${C_DIM}• Quick Submission Guide:${C_RESET}"
echo -e "  • ${C_BOLD}Standard CPU (<= 24h):${C_RESET}     sbatch -p common --qos=normal -t 24:00:00 --cpus-per-task=N --mem=XG ..."
echo -e "  • ${C_BOLD}Fast / Debug CPU (<= 2h):${C_RESET}    sbatch -p dedicated --qos=fast -t 02:00:00 ... ${C_DIM}(Starts immediately, high priority tier 5000)${C_RESET}"
echo -e "  • ${C_BOLD}Standard GPU (<= 3 days):${C_RESET}   sbatch -p gpu --qos=gpu --gres=gpu:1 -t 3-00:00:00 ..."
echo -e "  • ${C_BOLD}Fast / Debug GPU (<= 2h):${C_RESET}   sbatch -p dedicatedgpu --qos=fast --gres=gpu:1 -t 02:00:00 ... ${C_DIM}(Starts immediately, tier 5000)${C_RESET}"
echo -e "  • ${C_BOLD}Long CPU (> 24h):${C_RESET}           sbatch -p long --qos=long -t 14-00:00:00 ... ${C_DIM}(Low priority, up to 365 days)${C_RESET}"
echo ""

# 8. Section 7: Summary & Diagnostic
echo -e "${C_BOLD}${C_BLUE}[7] DIAGNOSTIC & PRIORITY SUMMARY${C_RESET}"
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
        printf "  1. %sMain Consumer in bis:%s %s%s%s accounts for %s~%.1f%%%s of group bis usage (~%.2f%% of entire cluster).\n", C_BOLD, C_RESET, C_RED, top_user, C_RESET, C_BOLD, top_group_pct, C_RESET, top_cluster_pct;
    } else {
        printf "  1. %sMain Consumer in bis:%s None (no active usage)\n", C_BOLD, C_RESET;
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
