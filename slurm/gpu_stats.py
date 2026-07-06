#!/usr/bin/env python3
import subprocess
import sys
import re
from collections import Counter, defaultdict

# GPU performance specs (FP32 TFLOPS) used to calculate speedup and sort inventory.
# 1. l40s: NVIDIA L40S (~91.6 TFLOPS)
# 2. A40: NVIDIA A40 (~37.4 TFLOPS)
# 3. A100: NVIDIA A100 (~19.5 TFLOPS FP32)
# 4. rtx6000: NVIDIA RTX 6000 (~16.3 TFLOPS)
# 5. 2g.48gb+gfx: MIG (Multi-Instance GPU) slice (~5.5 TFLOPS)
GPU_FP32_TFLOPS = {
    'l40s': 91.6,
    'A40': 37.4,
    'A100': 19.5,
    'rtx6000': 16.3,
    '2g.48gb+gfx': 5.5
}

GPU_MEM = {
    'l40s': '48GB',
    'A40': '48GB',
    'A100': '40GB/80GB',
    'rtx6000': '24GB',
    '2g.48gb+gfx': '48GB slice'
}





def run_ssh_command(cmd):
    try:
        res = subprocess.run(
            ["ssh", "maestro", cmd],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command over SSH: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def parse_nodes(sinfo_out):
    # Parses sinfo output format: NodeList|State|Gres
    gpu_inventory = defaultdict(int)
    node_states = Counter()
    
    for line in sinfo_out.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 3:
            continue
        nodelist, state, gres = parts[0], parts[1], parts[2]
        
        # Count node states
        state_clean = state.rstrip('*').rstrip('-')
        
        # We need to expand hostlists (e.g. maestro-[3011-3012,3015-3017] or maestro-3020)
        # To get the count of nodes
        node_count = 1
        match = re.search(r'\[([^\]]+)\]', nodelist)
        if match:
            elements = match.group(1).split(',')
            node_count = 0
            for elem in elements:
                if '-' in elem:
                    start, end = map(int, elem.split('-'))
                    node_count += (end - start + 1)
                else:
                    node_count += 1
        
        node_states[state_clean] += node_count
        
        # Parse GPUs in Gres
        # E.g. disk:890000M,gmem:no_consume:48G,gpu:A40:8,spread:9600
        # E.g. gpu:rtx6000:2,gpu:2g.48gb+gfx:12
        gpu_matches = re.findall(r'gpu:([^:]+):(\d+)', gres)
        for gpu_model, count in gpu_matches:
            gpu_inventory[gpu_model] += int(count) * node_count
            
    return gpu_inventory, node_states

def parse_queue(squeue_out):
    # Parses squeue output format: JobID|User|State|Reason|Gres(TRES)
    running_jobs = 0
    pending_jobs = 0
    gpus_in_use = defaultdict(int)
    pending_reasons = Counter()
    pending_users = Counter()
    
    for line in squeue_out.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 5:
            continue
        jobid, user, state, reason, gres = parts[0], parts[1], parts[2], parts[3], parts[4]
        
        if state == 'RUNNING':
            running_jobs += 1
            # Parse number of GPUs allocated to running jobs
            # E.g. gres/gpu:1 or gres/gpu:A100:2,gres/gmem:48G
            for item in gres.split(','):
                if 'gpu' in item and 'gmem' not in item:
                    parts_item = item.split(':')
                    if parts_item:
                        try:
                            gpu_count = int(parts_item[-1])
                            gpus_in_use['total'] += gpu_count
                            
                            # Check if specific GPU model is specified
                            if len(parts_item) > 2:
                                model = parts_item[1]
                                gpus_in_use[model] += gpu_count
                        except ValueError:
                            pass
        elif state == 'PENDING':
            pending_jobs += 1
            pending_reasons[reason] += 1
            pending_users[user] += 1
            
    return running_jobs, pending_jobs, gpus_in_use, pending_reasons, pending_users

def main():
    print("Fetching GPU cluster stats from Maestro...")
    sinfo_out = run_ssh_command("sinfo -p gpu -h -o '%N|%T|%G'")
    squeue_out = run_ssh_command("squeue -p gpu -h -o '%i|%u|%T|%r|%b'")
    
    gpu_inventory, node_states = parse_nodes(sinfo_out)
    running_jobs, pending_jobs, gpus_in_use, pending_reasons, pending_users = parse_queue(squeue_out)
    
    print("\n" + "="*50)
    print("             MAESTRO GPU DASHBOARD")
    print("="*50)
    
    print("\n[Node Status Summary]")
    for state, count in sorted(node_states.items()):
        print(f"  • {state.capitalize():<12}: {count} node(s)")
        
    print("\n[GPU Inventory] (sorted by GPU speed, fastest first)")
    total_gpus = 0
    # Sort by FP32 TFLOPS descending, and alphabetically for ties
    sorted_inventory = sorted(
        gpu_inventory.items(),
        key=lambda x: (GPU_FP32_TFLOPS.get(x[0], 0.0), x[0]),
        reverse=True
    )
    baseline_perf = GPU_FP32_TFLOPS.get('rtx6000', 16.3)
    for model, count in sorted_inventory:
        perf = GPU_FP32_TFLOPS.get(model, 0.0)
        speedup = perf / baseline_perf if baseline_perf else 0.0
        mem = GPU_MEM.get(model, 'unknown VRAM')
        if model == 'rtx6000':
            desc = f"{mem}, 1.00x baseline"
        elif 'slice' in model or model == '2g.48gb+gfx':
            desc = f"{mem}, {speedup:.2f}x slice"
        else:
            desc = f"{mem}, {speedup:.2f}x speedup"
        print(f"  • {model:<12} ({desc:<24}): {count} total")
        total_gpus += count

    print(f"  • Total GPUs  : {total_gpus}")


    
    print("\n[Queue Load]")
    print(f"  • Running Jobs: {running_jobs}")
    print(f"  • Pending Jobs: {pending_jobs}")
    print(f"  • GPUs Active : {gpus_in_use['total']} / {total_gpus} ({gpus_in_use['total']/total_gpus*100:.1f}% utilization)")
    
    if pending_jobs > 0:
        print("\n[Top Pending Reasons]")
        for reason, count in pending_reasons.most_common(5):
            print(f"  • {reason:<20}: {count} job(s)")
            
        print("\n[Top Users Waiting]")
        for user, count in pending_users.most_common(5):
            print(f"  • {user:<20}: {count} job(s)")
    print("="*50)

if __name__ == "__main__":
    main()
