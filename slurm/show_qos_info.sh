#!/usr/bin/env bash

# If we are not running on maestro, run the script remotely on maestro via SSH
if [[ "$(hostname)" != *maestro* ]]; then
    ssh maestro "/usr/bin/bash -s" < "$0" "$@"
    exit $?
fi

echo "=========================================================================="
echo "                         MAESTRO QUEUE & QoS INFO"
echo "=========================================================================="

echo ""
echo "[1] Available GPU Partitions & Nodes:"
echo "--------------------------------------------------------------------------"
# Move GRES to the end so it can expand dynamically without being truncated
sinfo -p gpu,gputest,dedicatedgpu -o "%20P %10T %20N %G"

echo ""
echo "[2] Configured QoS Limits (Run Times):"
echo "--------------------------------------------------------------------------"
# Use printf to format the limits cleanly
sacctmgr show qos format=name,MaxWall -p | tr '|' '\t' | grep -E "ultrafast|fast|normal|gpu" | while read -r qos_name time_limit; do
    if [ ! -z "$qos_name" ]; then
        printf "  • %-12s : %s\n" "$qos_name" "$time_limit"
    fi
done

echo ""
echo "[3] Your Authorized QoS Accounts:"
echo "--------------------------------------------------------------------------"
# Skip the header line (tail -n +2) to prevent printing 'User' as a config entry
sacctmgr show associations user=$USER format=user,account,partition,qos -p | tail -n +2 | while IFS='|' read -r user account partition qos dummy; do
    if [ ! -z "$user" ]; then
        echo "  • User     : $user"
        echo "  • Account  : $account"
        echo "  • Partition: ${partition:-All Default}"
        echo "  • Allowed QoS Tiers:"
        echo "$qos" | tr ',' '\n' | sed 's/^/      - /'
        echo "--------------------------------------------------------------------------"
    fi
done
echo "=========================================================================="
