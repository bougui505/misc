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

usage() {
    cat << "EOF"
Usage: maestro_bis_usage.sh [OPTIONS] [SECTION...]

Options:
  -s, --section SEC    Output only specified section(s) (comma or space-separated)
  -h, --help           Show this help message and exit

Available Sections:
  1, overview          Group BIS FairShare & consumption overview
  2, top10, groups     Top 10 groups cluster consumption ranking
  3, users             Per-user consumption in group BIS
  4, timing, myjobs    Your running jobs & task timing statistics
  5, livebis           Live jobs queued in group BIS
  6, globaljobs        Live jobs in common, gpu, dedicatedgpu (Cluster-wide)
  7, partitions        Accessible partitions & hardware characteristics
  8, waittime, queue   Expected queue wait time & live resource availability
  9, advice            Real-time partition selection advice
  10, diagnostic       Diagnostic & priority summary
  all (default)        Show all sections

Examples:
  maestro_bis_usage.sh                 # Show full dashboard (default)
  maestro_bis_usage.sh 8               # Show expected queue wait times
  maestro_bis_usage.sh waittime        # Show expected wait times by alias
  maestro_bis_usage.sh -s 1,4,8        # Show sections 1, 4, and 8
EOF
}

normalize_section() {
    local val=$(echo "$1" | tr "[:upper:]" "[:lower:]" | tr "," " ")
    local result=""
    for item in $val; do
        case $item in
            1|overview|group|fs)           result="$result 1" ;;
            2|top|top10|groups)            result="$result 2" ;;
            3|users|members)               result="$result 3" ;;
            4|timing|userjobs|myjobs|tasks) result="$result 4" ;;
            5|livebis|bisjobs)             result="$result 5" ;;
            6|globaljobs|clusterjobs)      result="$result 6" ;;
            7|partitions|specs|hardware)   result="$result 7" ;;
            8|waittime|wait|queue|avail)   result="$result 8" ;;
            9|advice|recommendations)      result="$result 9" ;;
            10|summary|diag|diagnostic)    result="$result 10" ;;
            all)                           result="all"; break ;;
            *) echo "Error: Unknown section '$item'" >&2; usage; exit 1 ;;
        esac
    done
    echo "$result"
}

SECTIONS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage; exit 0
            ;;
        -s|--section)
            if [[ -z "$2" ]]; then
                echo "Error: -s/--section requires an argument." >&2; exit 1
            fi
            parsed=$(normalize_section "$2")
            SECTIONS="$SECTIONS $parsed"
            shift 2
            ;;
        *)
            parsed=$(normalize_section "$1")
            SECTIONS="$SECTIONS $parsed"
            shift
            ;;
    esac
done

# Default to all if no section was specified
SECTIONS=$(echo "$SECTIONS" | xargs)
if [[ -z "$SECTIONS" ]]; then
    SECTIONS="all"
fi

show_sec() {
    [[ "$SECTIONS" == "all" || " $SECTIONS " =~ " $1 " ]]
}

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
squeue -A bis -h -r -o "%u|%P|%t|%r" 2>/dev/null
echo "===SQUEUE_ALL==="
squeue -p common,dedicated,gpu,dedicatedgpu,long,clcgwb -h -o "%P|%t|%r|%q|%u|%b" 2>/dev/null
echo "===SQUEUE_USER==="
squeue -u ${USER:-bougui} -h -o "%i|%j|%P|%t|%M|%N" 2>/dev/null
echo "===SINFO_NODES==="
sinfo -p common,dedicated,gpu,dedicatedgpu,long,clcgwb -h -N -o "%P|%N|%T|%C|%G" 2>/dev/null
echo "===SCONTROL_NODES==="
scontrol show nodes maestro-[3002-3009,3010-3020,3444-3451] 2>/dev/null | awk '/NodeName=/ {split($1, np, "="); node=np[2]} /AvailableFeatures=/ {feat=$0} /Gres=/ {gres=$0} /AllocTRES=/ {alloc=$0; print node "|" feat "|" gres "|" alloc}'
REMOTE
)

halflife=$(echo "$raw_dump" | sed -n '1p')
sshare_all_data=$(echo "$raw_dump" | sed -n '/===SSHARE_ALL===/,/===SSHARE_BIS===/{ /===SSHARE_ALL===/d; /===SSHARE_BIS===/d; p }')
sshare_data=$(echo "$raw_dump" | sed -n '/===SSHARE_BIS===/,/===SQUEUE_BIS===/{ /===SSHARE_BIS===/d; /===SQUEUE_BIS===/d; p }')
squeue_data=$(echo "$raw_dump" | sed -n '/===SQUEUE_BIS===/,/===SQUEUE_ALL===/{ /===SQUEUE_BIS===/d; /===SQUEUE_ALL===/d; p }')
squeue_all_data=$(echo "$raw_dump" | sed -n '/===SQUEUE_ALL===/,/===SQUEUE_USER===/{ /===SQUEUE_ALL===/d; /===SQUEUE_USER===/d; p }')
squeue_user_data=$(echo "$raw_dump" | sed -n '/===SQUEUE_USER===/,/===SINFO_NODES===/{ /===SQUEUE_USER===/d; /===SINFO_NODES===/d; p }')
sinfo_nodes_data=$(echo "$raw_dump" | sed -n '/===SINFO_NODES===/,/===SCONTROL_NODES===/{ /===SINFO_NODES===/d; /===SCONTROL_NODES===/d; p }')
scontrol_nodes_data=$(echo "$raw_dump" | sed -n '/===SCONTROL_NODES===/,$ { /===SCONTROL_NODES===/d; p }')

# 2. Section 1: Overview
if show_sec 1; then
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
fi

# 3. Section 2: Top 10 groups table
if show_sec 2; then
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
fi

# 4. Section 3: Per-user table for group bis
if show_sec 3; then
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
fi

# 5. Section 4: Your Running Jobs & Task Timing Statistics
if show_sec 4; then
echo -e "${C_BOLD}${C_BLUE}[4] YOUR RUNNING JOBS & TASK TIMING STATISTICS (${USER:-bougui})${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

echo "$squeue_user_data" | awk -F'|' -v C_BOLD="$C_BOLD" -v C_RESET="$C_RESET" -v C_GREEN="$C_GREEN" -v C_YELLOW="$C_YELLOW" '
function time_to_sec(t,   days, parts, n, d_split) {
    days = 0;
    if (index(t, "-") > 0) {
        split(t, d_split, "-");
        days = d_split[1] + 0;
        t = d_split[2];
    }
    n = split(t, parts, ":");
    if (n == 2) {
        return (days * 86400) + (parts[1] * 60) + parts[2];
    } else if (n == 3) {
        return (days * 86400) + (parts[1] * 3600) + (parts[2] * 60) + parts[3];
    }
    return 0;
}

function sec_to_time(s,   days, hours, mins, secs) {
    s = int(s + 0.5);
    days = int(s / 86400);
    s = s % 86400;
    hours = int(s / 3600);
    s = s % 3600;
    mins = int(s / 60);
    secs = s % 60;
    if (days > 0) {
        return sprintf("%d-%02d:%02d:%02d", days, hours, mins, secs);
    } else if (hours > 0) {
        return sprintf("%02d:%02d:%02d", hours, mins, secs);
    } else {
        return sprintf("%02d:%02d", mins, secs);
    }
}

NF>=5 {
    jobid = $1;
    name = $2;
    part = $3;
    st = $4;
    timestr = $5;
    node = $6;
    sec = time_to_sec(timestr);
    
    if (st == "R") {
        r_tasks++;
        r_sec_total += sec;
        
        task_list[r_tasks] = sprintf("%s|%s|%s|%s|%s", jobid, name, part, timestr, node);
        
        if (r_tasks == 1 || sec > max_sec) {
            max_sec = sec;
            max_task = jobid;
            max_name = name;
            max_node = node;
        }
        if (r_tasks == 1 || sec < min_sec) {
            min_sec = sec;
            min_task = jobid;
        }
    } else if (st == "PD") {
        pd_tasks++;
    }
}

END {
    if (r_tasks == 0 && pd_tasks == 0) {
        printf "  %sNo active or pending jobs found for user.%s\n\n", C_GREEN, C_RESET;
        exit;
    }
    
    if (r_tasks == 0) {
        printf "  • Currently Running Tasks : %s0%s (%d pending jobs queued)\n\n", C_BOLD, C_RESET, pd_tasks;
        exit;
    }
    
    avg_sec = r_sec_total / r_tasks;
    
    printf "  • Currently Running Tasks : %s%s%d task(s)%s (plus %d pending jobs queued)\n", C_BOLD, C_GREEN, r_tasks, C_RESET, pd_tasks;
    printf "  • Average Task Runtime    : %s%s%s (%d sec)\n", C_BOLD, sec_to_time(avg_sec), C_RESET, int(avg_sec);
    printf "  • Max Observed Runtime    : %s%s%s%s (Task: %s, Name: %s, Node: %s)\n", C_BOLD, C_YELLOW, sec_to_time(max_sec), C_RESET, max_task, max_name, max_node;
    printf "  • Min Observed Runtime    : %s%s%s (Task: %s)\n\n", C_BOLD, sec_to_time(min_sec), C_RESET, min_task;
    
    print "TASK_ID|JOB_NAME|PARTITION|RUNTIME|NODE";
    for (i = 1; i <= r_tasks; i++) {
        print task_list[i];
    }
}
' | column -t -s '|'
echo ""
fi

# 6. Section 5: Live Jobs in Group BIS
if show_sec 5; then
echo -e "${C_BOLD}${C_BLUE}[5] LIVE JOBS IN GROUP BIS${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

if [ -z "$squeue_data" ]; then
    echo -e "  ${C_GREEN}No active or pending jobs currently queued for account bis.${C_RESET}"
else
    (
      echo -e "USER|PARTITION|STATE|COUNT|REASON"
      echo "$squeue_data" | awk -F'|' '
      NF>=4 {
          user = $1; gsub(/^[ \t]+|[ \t]+$/, "", user);
          part = $2; gsub(/^[ \t]+|[ \t]+$/, "", part);
          state = $3; gsub(/^[ \t]+|[ \t]+$/, "", state);
          reason = $4; gsub(/^[ \t]+|[ \t]+$/, "", reason);
          key = user "|" part "|" state "|" reason;
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
fi

# 7. Section 6: Live Jobs in Common, GPU & DedicatedGPU Partitions (Global)
if show_sec 6; then
echo -e "${C_BOLD}${C_BLUE}[6] LIVE JOBS IN COMMON, GPU & DEDICATEDGPU PARTITIONS (Global)${C_RESET}"
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
fi

# 8. Section 7: Accessible Partitions for Account BIS
if show_sec 7; then
echo -e "${C_BOLD}${C_BLUE}[7] ACCESSIBLE PARTITIONS & CHARACTERISTICS FOR BIS${C_RESET}"
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
echo -e "  • ${C_BOLD}Standard GPU (<= 3 days):${C_RESET}   sbatch -p gpu --qos=gpu --gres=gpu:1 -C \"sm_80|sm_86|sm_89|sm_120\" -t 3-00:00:00 ..."
echo -e "  • ${C_BOLD}Fast / Debug GPU (<= 2h):${C_RESET}   sbatch -p dedicatedgpu --qos=fast --gres=gpu:1 -C \"sm_80|sm_86|sm_89|sm_120\" -t 02:00:00 ..."
echo -e "  • ${C_BOLD}Long CPU (> 24h):${C_RESET}           sbatch -p long --qos=long -t 14-00:00:00 ... ${C_DIM}(Low priority, up to 365 days)${C_RESET}"
echo -e "  • ${C_BOLD}Update Pending Job GPU:${C_RESET}   scontrol update JobId=<JOBID> Features=\"sm_80|sm_86|sm_89|sm_120\""
echo ""
fi

# 9. Section 8: Expected Queue Wait Time & Live Availability
if show_sec 8; then
echo -e "${C_BOLD}${C_BLUE}[8] EXPECTED QUEUE WAIT TIME & RESOURCE AVAILABILITY (${USER:-bougui})${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

{
  echo "===SSHARE==="
  echo "$sshare_data"
  echo "===SCTRL==="
  echo "$scontrol_nodes_data"
  echo "===SINFO==="
  echo "$sinfo_nodes_data"
  echo "===SQUEUE==="
  echo "$squeue_all_data"
} | awk -v C_BOLD="$C_BOLD" -v C_RESET="$C_RESET" -v C_GREEN="$C_GREEN" -v C_YELLOW="$C_YELLOW" -v C_RED="$C_RED" -v C_CYAN="$C_CYAN" '
BEGIN {
    mode = "";
}
/^===SSHARE===$/ { mode="sshare"; next }
/^===SINFO===$/ { mode="sinfo"; next }
/^===SCTRL===$/ { mode="sctrl"; next }
/^===SQUEUE===$/ { mode="squeue"; next }

mode == "sctrl" && NF {
    split($0, p, "|");
    node = p[1]; feat = p[2]; gres = p[3]; alloc = p[4];
    
    is_sm120 = (feat ~ /sm_120/ || feat ~ /RTX6000/);
    node_is_sm120[node] = is_sm120;
    
    n = split(gres, gitems, ",");
    for (j=1; j<=n; j++) {
        if (gitems[j] ~ /gpu:/) {
            split(gitems[j], gp, ":");
            c = gp[length(gp)]; gsub(/[^0-9]/, "", c);
            node_tot_gpu[node] += c;
        }
    }
    n = split(alloc, aitems, ",");
    for (j=1; j<=n; j++) {
        if (aitems[j] ~ /gres\/gpu=/) {
            split(aitems[j], ap, "=");
            c = ap[2]; gsub(/[^0-9]/, "", c);
            node_alloc_gpu[node] += c;
        }
    }
    next;
}

mode == "sinfo" && NF {
    split($0, p, "|");
    part = p[1]; gsub(/\*/, "", part);
    node = p[2]; state = p[3]; cpus = p[4];
    split(cpus, cp, "/");
    p_tot_cpu[part] += cp[4];
    p_idle_cpu[part] += cp[2];
    p_tot_nodes[part]++;
    if (state ~ /idle/) p_idle_nodes[part]++;
    
    if (node in node_tot_gpu && !seen_node[part, node]) {
        seen_node[part, node] = 1;
        tot_g = node_tot_gpu[node];
        alloc_g = node_alloc_gpu[node];
        free_g = tot_g - alloc_g;
        if (free_g < 0) free_g = 0;
        
        p_tot_gpu[part] += tot_g;
        p_alloc_gpu[part] += alloc_g;
        
        if (node_is_sm120[node]) {
            p_free_sm120_gpu[part] += free_g;
            p_tot_sm120_gpu[part] += tot_g;
        } else {
            p_free_std_gpu[part] += free_g;
            p_tot_std_gpu[part] += tot_g;
        }
    }
    next;
}

mode == "squeue" && NF {
    split($0, p, "|");
    part = p[1]; state = p[2]; reason = p[3]; qos = p[4];
    if (state == "PD") p_pd_jobs[part]++;
    else if (state == "R") p_r_jobs[part]++;
    next;
}

mode == "sshare" && NF {
    split($0, cols, "|");
    if (cols[1] == "bis" && (cols[2] == "" || cols[2] ~ /^[ \t]*$/)) {
        target = cols[3] + 0;
        usage = cols[5] + 0;
        fs_ratio = (target > 0 ? usage / target : 0);
    }
    next;
}

END {
    # Output Table
    print "PARTITION|BEST_QOS|FREE_RESOURCE|QUEUE_DEPTH|ESTIMATED_DISPATCH|COMMENTS";

    # dedicated (CPU)
    free_cpu = p_idle_cpu["dedicated"] + 0;
    pd_cnt = p_pd_jobs["dedicated"] + 0;
    nodes_cnt = p_idle_nodes["dedicated"] + 0;
    if (nodes_cnt > 0) {
        res_raw = sprintf("%'"'"'d idle cores (%d fully empty nodes)", free_cpu, nodes_cnt);
    } else {
        res_raw = sprintf("%'"'"'d idle cores", free_cpu);
    }
    if (free_cpu == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else if (free_cpu < 16) {
        res_str = sprintf("%s%s%s", C_YELLOW, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }

    if (free_cpu >= 8 && pd_cnt == 0) {
        est = sprintf("%sImmediate (< 1 min)%s", C_GREEN, C_RESET);
        comment = "Plentiful opportunistic CPU slots (Tier 5000)";
    } else if (free_cpu >= 8) {
        est = sprintf("%sFast (~1-3 min)%s", C_GREEN, C_RESET);
        comment = sprintf("Tier 5000 bypasses %d lower tier pending jobs", pd_cnt);
    } else {
        est = sprintf("%sShort (~5-15 min)%s", C_YELLOW, C_RESET);
        comment = "Waiting for short running task completions";
    }
    printf "dedicated|fast / ultrafast|%s|%d PD|%s|%s\n", res_str, pd_cnt, est, comment;

    # dedicatedgpu (GPU)
    tot_dgpu = p_tot_gpu["dedicatedgpu"] + 0;
    alloc_dgpu = p_alloc_gpu["dedicatedgpu"] + 0;
    free_std = p_free_std_gpu["dedicatedgpu"] + 0;
    tot_std = p_tot_std_gpu["dedicatedgpu"] + 0;
    free_sm120 = p_free_sm120_gpu["dedicatedgpu"] + 0;
    tot_sm120 = p_tot_sm120_gpu["dedicatedgpu"] + 0;
    pd_dgpu = p_pd_jobs["dedicatedgpu"] + 0;
    
    res_raw = sprintf("%d/%d free A100/A40, %d/%d free RTX6000", free_std, tot_std, free_sm120, tot_sm120);
    if (free_std + free_sm120 == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else if (free_std == 0) {
        res_str = sprintf("%s%s%s", C_YELLOW, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }

    if (free_std > 0) {
        est = sprintf("%sImmediate (< 1 min)%s", C_GREEN, C_RESET);
        comment = "Free A100/A40 & RTX6000 slots available now";
    } else if (free_sm120 > 0) {
        est = sprintf("%sImmediate with -C sm_120%s", C_GREEN, C_RESET);
        comment = sprintf("A100/A40 full; %d free RTX6000 Ada (see tip below to update)", free_sm120);
    } else {
        est = sprintf("%sShort (~5-25 min)%s", C_YELLOW, C_RESET);
        comment = sprintf("All 9 dedicated nodes busy (%d PD ahead, Tier 5000)", pd_dgpu);
    }
    printf "dedicatedgpu|fast / ultrafast|%s|%d PD|%s|%s\n", res_str, pd_dgpu, est, comment;

    # common (CPU)
    free_comm = p_idle_cpu["common"] + 0;
    pd_comm = p_pd_jobs["common"] + 0;
    nodes_comm = p_idle_nodes["common"] + 0;
    if (nodes_comm > 0) {
        res_raw = sprintf("%'"'"'d idle cores (%d fully empty nodes)", free_comm, nodes_comm);
    } else {
        res_raw = sprintf("%'"'"'d idle cores", free_comm);
    }
    if (free_comm == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else if (free_comm < 32) {
        res_str = sprintf("%s%s%s", C_YELLOW, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }

    if (free_comm >= 8 && pd_comm <= 5) {
        est = sprintf("%sShort (< 5 min)%s", C_GREEN, C_RESET);
        comment = "Standard cluster CPU pool has idle cores";
    } else if (free_comm >= 8) {
        est = sprintf("%sModerate (~5-20 min)%s", C_YELLOW, C_RESET);
        comment = sprintf("%d pending jobs in common queue", pd_comm);
    } else {
        est = sprintf("%sDelayed (~30m-2h)%s", C_RED, C_RESET);
        comment = "Common CPU pool congested; FairShare evaluated";
    }
    printf "common|normal (<=24h)|%s|%d PD|%s|%s\n", res_str, pd_comm, est, comment;

    # gpu (Standard GPU pool)
    tot_gpu = p_tot_gpu["gpu"] + 0;
    alloc_gpu = p_alloc_gpu["gpu"] + 0;
    free_std_g = p_free_std_gpu["gpu"] + 0;
    tot_std_g = p_tot_std_gpu["gpu"] + 0;
    free_sm120_g = p_free_sm120_gpu["gpu"] + 0;
    tot_sm120_g = p_tot_sm120_gpu["gpu"] + 0;
    pd_gpu = p_pd_jobs["gpu"] + 0;
    
    res_raw = sprintf("%d/%d free A100/A40, %d/%d free RTX6000", free_std_g, tot_std_g, free_sm120_g, tot_sm120_g);
    if (free_std_g + free_sm120_g == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else if (free_std_g == 0 || free_std_g < 4) {
        res_str = sprintf("%s%s%s", C_YELLOW, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }

    if (free_std_g >= 2 && pd_gpu == 0) {
        est = sprintf("%sShort (~5-15 min)%s", C_GREEN, C_RESET);
        comment = "Free standard GPU slots available";
    } else if (free_std_g > 0 || free_sm120_g > 0) {
        est = sprintf("%sModerate (~15m-1h)%s", C_YELLOW, C_RESET);
        comment = sprintf("%d pending GPU jobs competing on priority", pd_gpu);
    } else {
        est = sprintf("%sHigh Wait (Hours - Days)%s", C_RED, C_RESET);
        if (fs_ratio > 2) {
            comment = sprintf("Saturated pool + %d PD + bis %.1fx over FairShare target", pd_gpu, fs_ratio);
        } else {
            comment = sprintf("Saturated pool + %d pending jobs in queue", pd_gpu);
        }
    }
    printf "gpu|gpu (<=3 days)|%s|%d PD|%s|%s\n", res_str, pd_gpu, est, comment;

    # long (Long CPU runs)
    free_long = p_idle_cpu["long"] + 0;
    pd_long = p_pd_jobs["long"] + 0;
    nodes_long = p_idle_nodes["long"] + 0;
    if (nodes_long > 0) {
        res_raw = sprintf("%d idle cores (%d fully empty nodes)", free_long, nodes_long);
    } else {
        res_raw = sprintf("%d idle cores", free_long);
    }
    if (free_long == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else if (free_long < 32) {
        res_str = sprintf("%s%s%s", C_YELLOW, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }

    if (free_long > 0) {
        est = sprintf("%sShort (~5-30 min)%s", C_GREEN, C_RESET);
        comment = "Slots available on dedicated 4 long nodes";
    } else {
        est = sprintf("%sVariable / Long%s", C_RED, C_RESET);
        comment = "All 4 long nodes fully allocated (up to 365d walltime)";
    }
    printf "long|long (>24h)|%s|%d PD|%s|%s\n", res_str, pd_long, est, comment;

    # clcgwb
    free_clc = p_idle_cpu["clcgwb"] + 0;
    nodes_clc = p_idle_nodes["clcgwb"] + 0;
    if (nodes_clc > 0) {
        res_raw = sprintf("%d idle cores (%d fully empty nodes)", free_clc, nodes_clc);
    } else {
        res_raw = sprintf("%d idle cores", free_clc);
    }
    if (free_clc == 0) {
        res_str = sprintf("%s%s%s", C_RED, res_raw, C_RESET);
    } else {
        res_str = sprintf("%s%s%s", C_GREEN, res_raw, C_RESET);
    }
    printf "clcgwb|normal|%s|%d PD|%sImmediate (< 1 min)%s|Specialized CLC Workbench node (Tier 10000)\n", res_str, p_pd_jobs["clcgwb"]+0, C_GREEN, C_RESET;
}
' | column -t -s '|'

echo -e "\n${C_DIM}• Tip: If pending on dedicatedgpu with free RTX6000 available, update via:${C_RESET}"
echo -e "  ${C_BOLD}scontrol update JobId=<JOBID> Features=\"sm_80|sm_86|sm_89|sm_120\"${C_RESET}"
echo ""
fi

# 10. Section 9: Real-Time Partition Selection Advice
if show_sec 9; then
echo -e "${C_BOLD}${C_BLUE}[9] REAL-TIME PARTITION SELECTION ADVICE${C_RESET}"
echo -e "${C_DIM}${SUBSEP}${C_RESET}"

echo "$squeue_all_data" | awk -F'|' -v sshare="$sshare_data" -v C_BOLD="$C_BOLD" -v C_RESET="$C_RESET" -v C_GREEN="$C_GREEN" -v C_YELLOW="$C_YELLOW" -v C_CYAN="$C_CYAN" -v C_DIM="$C_DIM" '
BEGIN {
    split(sshare, srows, "\n");
    for (i in srows) {
        split(srows[i], cols, "|");
        if (cols[1] == "bis" && (cols[2] == "" || cols[2] ~ /^[ \t]*$/)) {
            target = cols[3] + 0;
            usage = cols[5] + 0;
            fs_ratio = (target > 0 ? usage / target : 0);
        }
    }
}
NF>=3 {
    part = $1;
    st = $2;
    if (st == "R") r[part]++;
    else if (st == "PD") pd[part]++;
}
END {
    printf "  %s1. Ultra-Short GPU Workloads (<= 5 min)%s %s[Single inference / Quick tests / Array batches]:%s\n", C_BOLD, C_RESET, C_DIM, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=dedicatedgpu --qos=ultrafast -t 00:05:00 --gres=gpu:1 -C \"sm_80|sm_86|sm_89|sm_120\"%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
    printf "     • %sWhy:%s Tier 5000 + %s+500 points QoS priority boost%s. Adding %ssm_120%s allows dispatching to idle RTX6000 Ada nodes.\n\n", C_BOLD, C_RESET, C_CYAN, C_RESET, C_CYAN, C_RESET;

    printf "  %s2. Short GPU Workloads (<= 2 hours)%s %s[Preprocessing / Fast fine-tuning / Debugging]:%s\n", C_BOLD, C_RESET, C_DIM, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=dedicatedgpu --qos=fast -t 02:00:00 --gres=gpu:1 -C \"sm_80|sm_86|sm_89|sm_120\"%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
    printf "     • %sWhy:%s PriorityTier is %s5000%s. Starts much faster on dedicated GPU pool.\n", C_BOLD, C_RESET, C_CYAN, C_RESET;
    printf "     • %sCurrent Load:%s %d running, %d pending on dedicatedgpu.\n\n", C_DIM, C_RESET, r["dedicatedgpu"]+0, pd["dedicatedgpu"]+0;

    printf "  %s3. Long GPU Workloads (> 2 hours, up to 3 days)%s %s[Large model training / Long simulations]:%s\n", C_BOLD, C_RESET, C_DIM, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=gpu --qos=gpu -t 3-00:00:00 --gres=gpu:1 -C \"sm_80|sm_86|sm_89|sm_120\"%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
    if (fs_ratio > 2) {
        printf "     • %sNotice:%s Group bis usage is %s%.1fx over target%s; standard gpu jobs will have lower FairShare priority.\n", C_YELLOW, C_RESET, C_YELLOW, fs_ratio, C_RESET;
        printf "       Current \"gpu\" queue has %s%d running, %d pending%s. Expect queue time before dispatch.\n\n", C_BOLD, r["gpu"]+0, pd["gpu"]+0, C_RESET;
    } else {
        printf "     • %sCurrent Load:%s %d running, %d pending on gpu partition.\n\n", C_DIM, C_RESET, r["gpu"]+0, pd["gpu"]+0;
    }

    printf "  %s4. Short CPU Workloads (<= 2 hours)%s %s[Compilation / Array tasks / Data parsing]:%s\n", C_BOLD, C_RESET, C_DIM, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=dedicated --qos=fast -t 02:00:00%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
    printf "     • %sWhy:%s PriorityTier %s5000%s on 112 opportunistic nodes (10,752 cores), avoiding common queue penalties.\n", C_BOLD, C_RESET, C_CYAN, C_RESET;
    printf "     • %sCurrent Load:%s %d running on dedicated pool.\n\n", C_DIM, C_RESET, r["dedicated"]+0;

    printf "  %s5. Standard CPU Workloads (2h - 24h)%s %s[Standard batch jobs]:%s\n", C_BOLD, C_RESET, C_DIM, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=common --qos=normal -t 24:00:00%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
    printf "     • %sCurrent Load:%s %d running, %d pending on common.\n\n", C_DIM, C_RESET, r["common"]+0, pd["common"]+0;

    printf "  %s6. Long CPU Workloads (> 24h, up to 365 days)%s:\n", C_BOLD, C_RESET;
    printf "     • %sBest Choice:%s %s--partition=long --qos=long -t 14-00:00:00%s\n", C_BOLD, C_RESET, C_GREEN, C_RESET;
}
'
echo ""
fi

# 11. Section 10: Summary & Diagnostic
if show_sec 10; then
echo -e "${C_BOLD}${C_BLUE}[10] DIAGNOSTIC & PRIORITY SUMMARY${C_RESET}"
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
echo ""
fi
echo -e "${C_BOLD}${C_CYAN}${SEP}${C_RESET}\n"

