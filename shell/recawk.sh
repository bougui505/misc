#!/usr/bin/env bash
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

cat << 'EOF' > "$MYTMP/helper.py"
import sys
import os
import time
import json

try:
    import duckdb
except ImportError:
    sys.stderr.write("Error: 'duckdb' Python package is required for Parquet support.\n")
    sys.stderr.write("Please install it using: pip install duckdb\n")
    sys.exit(1)

def open_file_decompressed(file_path):
    if file_path == '-':
        import sys
        return sys.stdin, None
    elif file_path.endswith('.gz'):
        import subprocess
        has_pigz = False
        try:
            has_pigz = subprocess.run(['which', 'pigz'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        except Exception:
            pass
        decompressor = 'pigz' if has_pigz else 'gzip'
        proc = subprocess.Popen([decompressor, '-dc', file_path], stdout=subprocess.PIPE, text=True, bufsize=262144, encoding='utf-8', errors='ignore')
        return proc.stdout, proc
    else:
        return open(file_path, 'r', encoding='utf-8', errors='ignore'), None

mode = sys.argv[1]

if mode == 'to':
    db_path = sys.argv[2]
    files = sys.argv[3:]
    
    valid_files = []
    for f in files:
        if f == '-' or os.path.exists(f):
            valid_files.append(f)
            
    class ProgressReporter:
        def __init__(self, files):
            self.files = files
            self.use_percentage = False
            self.total_records = 0
            self.record_count = 0
            self.last_update_time = 0
            self.start_time = time.time()
            
            if files and all(f != '-' for f in files):
                try:
                    total_est_records = 0
                    for f in files:
                        file_size = os.path.getsize(f)
                        if file_size > 0:
                            if f.endswith('.gz'):
                                import subprocess
                                # Decompress a small sample of the Gzip file to count density
                                proc = subprocess.Popen(['gzip', '-dc', f], stdout=subprocess.PIPE, text=True, bufsize=262144, errors='ignore')
                                sample_lines = []
                                decomp_bytes = 0
                                for _ in range(5000):
                                    line = proc.stdout.readline()
                                    if not line:
                                        break
                                    sample_lines.append(line)
                                    decomp_bytes += len(line)
                                proc.terminate()
                                proc.wait()
                                
                                sample_records = sum(1 for l in sample_lines if l.rstrip('\r\n') == "--")
                                if sample_records > 0 and decomp_bytes > 0:
                                    total_est_records += int((file_size * 8) * (sample_records / decomp_bytes))
                            else:
                                # Sample first 500 KB of raw file
                                sample_size = min(500 * 1024, file_size)
                                with open(f, 'r', encoding='utf-8', errors='ignore') as sample_file:
                                    sample_lines = sample_file.readlines(sample_size)
                                    decomp_bytes = sum(len(l) for l in sample_lines)
                                    sample_records = sum(1 for l in sample_lines if l.rstrip('\r\n') == "--")
                                    if sample_records > 0 and decomp_bytes > 0:
                                        total_est_records += int(file_size * (sample_records / decomp_bytes))
                    self.total_records = total_est_records
                    if self.total_records > 0:
                        self.use_percentage = True
                except Exception:
                    self.use_percentage = False

        def update(self, line_len=0, is_record=False):
            if is_record:
                self.record_count += 1
            
            now = time.time()
            if now - self.last_update_time >= 0.2:
                self.print_progress()
                self.last_update_time = now

            if is_record and self.record_count % 50000 == 0:
                self.print_progress()
                self.last_update_time = now

        def print_progress(self):
            if '--silent' in sys.argv:
                return
            sys.stderr.write("\r\033[K")
            if self.use_percentage:
                pct = (self.record_count / self.total_records) * 100
                pct = min(99.9, pct)
                bar_width = 30
                filled = int(bar_width * pct / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                # Calculate ETA
                now = time.time()
                elapsed = now - self.start_time
                if elapsed > 0.5 and self.record_count > 0:
                    rate = self.record_count / elapsed
                    rem_records = max(0, self.total_records - self.record_count)
                    eta_sec = int(rem_records / rate)
                    if eta_sec < 60:
                        eta_str = f"{eta_sec}s"
                    elif eta_sec < 3600:
                        m = eta_sec // 60
                        s = eta_sec % 60
                        eta_str = f"{m}m {s:02d}s"
                    else:
                        h = eta_sec // 3600
                        m = (eta_sec % 3600) // 60
                        eta_str = f"{h}h {m:02d}m"
                else:
                    eta_str = "--"
                    
                sys.stderr.write(f"Converting to DB: [{bar}] {pct:.1f}% ({self.record_count:,}/{self.total_records:,} records) | ETA: {eta_str}")
            else:
                sys.stderr.write(f"Converting to DB: {self.record_count:,} records processed...")
            sys.stderr.flush()

        def print_final_progress(self):
            if '--silent' in sys.argv:
                return
            sys.stderr.write("\r\033[K")
            bar_width = 30
            bar = '█' * bar_width
            sys.stderr.write(f"Converting to DB: [{bar}] 100.0% ({self.record_count:,}/{self.record_count:,} records)")
            sys.stderr.flush()

        def finish(self):
            if '--silent' in sys.argv:
                return
            if self.use_percentage:
                self.print_final_progress()
            else:
                self.print_progress()
            sys.stderr.write("\nDone!\n")
            sys.stderr.flush()

    chunk_size = 100000
    temp_dir = db_path + ".tmp_parts"
    os.makedirs(temp_dir, exist_ok=True)
    temp_jsonl_path = os.path.join(temp_dir, "temp.jsonl")
    temp_parquet_files = []
    
    current_records = []
    count = 0
    part_idx = 0
    
    def flush_chunk_to_parquet(records_list):
        global part_idx
        if not records_list:
            return
        with open(temp_jsonl_path, 'w', encoding='utf-8', errors='ignore') as jsonl_file:
            for r in records_list:
                jsonl_file.write(json.dumps(r) + '\n')
                
        part_path = os.path.join(temp_dir, f"part_{part_idx}.parquet")
        temp_parquet_files.append(part_path)
        
        conn = duckdb.connect(':memory:')
        conn.execute(f"COPY (SELECT * FROM read_json_auto('{temp_jsonl_path}')) TO '{part_path}' (FORMAT 'PARQUET')")
        conn.close()
        
        if os.path.exists(temp_jsonl_path):
            try: os.remove(temp_jsonl_path)
            except: pass
            
        records_list.clear()
        part_idx += 1

    reporter = ProgressReporter(valid_files)
    current_record = {}
    
    if not valid_files:
        for line in sys.stdin:
            line_len = len(line)
            line_clean = line.rstrip('\r\n')
            if line_clean == "--":
                if current_record:
                    current_record['rowid'] = count + 1
                    current_records.append(dict(current_record))
                    current_record.clear()
                    count += 1
                    reporter.update(line_len, is_record=True)
                    if len(current_records) >= chunk_size:
                        flush_chunk_to_parquet(current_records)
                else:
                    reporter.update(line_len, is_record=False)
            else:
                if '=' in line_clean:
                    parts = line_clean.split('=', 1)
                    current_record[parts[0]] = parts[1]
                reporter.update(line_len, is_record=False)
    else:
        for f in valid_files:
            fh, proc = open_file_decompressed(f)
            try:
                for line in fh:
                    line_len = len(line)
                    line_clean = line.rstrip('\r\n')
                    if line_clean == "--":
                        if current_record:
                            current_record['rowid'] = count + 1
                            current_records.append(dict(current_record))
                            current_record.clear()
                            count += 1
                            reporter.update(line_len, is_record=True)
                            if len(current_records) >= chunk_size:
                                flush_chunk_to_parquet(current_records)
                        else:
                            reporter.update(line_len, is_record=False)
                    else:
                        if '=' in line_clean:
                            parts = line_clean.split('=', 1)
                            current_record[parts[0]] = parts[1]
                        reporter.update(line_len, is_record=False)
            finally:
                if f != '-':
                    fh.close()
                if proc:
                    proc.terminate()
                    proc.wait()
                    
    if current_record:
        current_record['rowid'] = count + 1
        current_records.append(dict(current_record))
        reporter.update(0, is_record=True)
        
    flush_chunk_to_parquet(current_records)
        
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass
        
    if temp_parquet_files:
        sys.stderr.write("\nWriting final Parquet file...\n")
        sys.stderr.flush()
        
        conn = duckdb.connect(':memory:')
        conn.execute(f"COPY (SELECT * FROM read_parquet({temp_parquet_files}, union_by_name=True)) TO '{db_path}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD')")
        conn.close()
        
    import shutil
    if os.path.exists(temp_dir):
        try: shutil.rmtree(temp_dir)
        except: pass
        
    reporter.finish()

elif mode == 'from':
    db_path = sys.argv[2]
    where_clause = sys.argv[3] if sys.argv[3] != 'NONE' else None
    used_fields_str = sys.argv[4] if len(sys.argv) > 4 else ''
    
    cols_to_query = [f.strip() for f in used_fields_str.split(',') if f.strip()] if used_fields_str else None
    
    conn = duckdb.connect(':memory:')
    
    try:
        cursor = conn.execute(f"DESCRIBE SELECT * FROM '{db_path}'")
        all_cols = [row[0] for row in cursor.fetchall() if row[0] != 'rowid']
    except Exception as e:
        sys.stderr.write(f"Error reading Parquet file: {e}\n")
        sys.exit(1)
        
    if cols_to_query:
        cols = [c for c in cols_to_query if c in all_cols]
    else:
        cols = all_cols
        
    if not cols:
        cursor = conn.execute(f"SELECT count(*) FROM '{db_path}'")
        count = cursor.fetchone()[0]
        for _ in range(count):
            print("--")
        sys.exit(0)
        
    col_selectors = ', '.join(f'"{c}"' for c in cols)
    query = f"SELECT {col_selectors} FROM '{db_path}'"
    if where_clause:
        query += f' WHERE {where_clause}'
        
    res = conn.execute(query)
    
    try:
        while True:
            rows = res.fetchmany(1000)
            if not rows:
                break
            for row in rows:
                for col_name, val in zip(cols, row):
                    if val is not None:
                        sys.stdout.write(f"{col_name}={val}\n")
                sys.stdout.write("--\n")
    except BrokenPipeError:
        pass
    finally:
        conn.close()
EOF

function usage () {
    cat << EOF
Usage: recawk [OPTIONS] 'AWK_SCRIPT' [FILES]

A powerful tool to process record-formatted files (key=value) using AWK.

Options:
  -h, --help           Print this help message and exit
  -n, --nrec           Print the number of records
  -e, --est-nrec       Estimate the number of records (for large files)
  -s, --sample N       Pick N random records (reservoir sampling)
  -k, --keys           Print all unique keys present in the file
  --torec SEP          Convert column-based files (e.g., CSV/TSV) to rec format
  --tocsv              Convert rec format to CSV (first record defines columns)
  -v VAR=VAL           Pass a variable to the AWK script
  --todb               Convert rec format to Parquet database (uses input file basename)
  -w, --where CLAUSE   SQL WHERE clause to filter records when querying a Parquet database

Record Format:
  Records are separated by '--' on a line by itself.
  Example:
    key1=val1
    key2=val2
    --
    key1=val3
    key2=val4
    --

AWK Integration:
  - Each record is loaded into the 'rec' associative array.
  - Access fields using: rec["key1"]
  - Current record count is available in 'nr' and 'fnr'.
  - Predefined functions:
      printrec()       - Print the current record in key=val format
      spearman(x,y,n)  - Compute Spearman correlation for arrays of length n
      pearson(x,y,n)   - Compute Pearson correlation for arrays of length n

Performance Optimizations:
  - Smart Filtering: recawk detects used fields (e.g., rec["tmscore"]) and pre-filters
    the input using 'grep' or SQL column-selects to significantly speed up processing.
  - Pair with pigz: For maximum speed on .gz files, use: pigz -dc file.rec.gz | recawk ...

Examples:
  # Extract a single field from a compressed file
  zcat data.rec.gz | recawk '{print rec["tmscore"]}'

  # Convert data to Parquet database (automatically outputs to data.parquet)
  recawk --todb data.rec

  # Query Parquet database seamlessly
  recawk '{print rec["tmscore"]}' data.parquet

  # Query Parquet database with database-level filtering
  recawk -w "tmscore > 0.8" '{print rec["tmscore"]}' data.parquet
EOF
}

V="V=0"
GETNREC=0
ESTNREC=0
SAMPLE=0
TOREC=0
KEYS=0
TOCSV=0
TODB=0
WHERE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -v) V=$2; shift 2 ;;
        -n|--nrec) GETNREC=1; shift ;;
        -e|--est-nrec) ESTNREC=1; shift ;;
        -s|--sample) SAMPLE=$2; shift 2 ;;
        --torec) TOREC=$2; shift 2 ;;
        --keys) KEYS=1; shift ;;
        --tocsv) TOCSV=1; shift ;;
        --todb) TODB=1; shift ;;
        -w|--where) WHERE=$2; shift 2 ;;
        *) break ;;
    esac
done

getnrec(){
    gzip -dfc "$1" 2>/dev/null | grep -c "^--$"
}

estnrec(){
    local file="$1"
    if [ -z "$file" ] || [ "$file" = "-" ]; then
        echo "Error: Estimation requires a file path, cannot estimate from stdin." >&2
        exit 1
    fi
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist." >&2
        exit 1
    fi

    local total_bytes
    total_bytes=$(stat -c %s "$file")

    # Sample 100 MiB or 10 MiB depending on file size
    local sample_size_mb=100
    if [ "$total_bytes" -lt $(( 100 * 1024 * 1024 )) ]; then
        sample_size_mb=10
    fi

    local sample_bytes=$(( sample_size_mb * 1024 * 1024 ))
    if [ "$total_bytes" -le "$sample_bytes" ]; then
        local lines
        lines=$(gzip -dfc "$file" | wc -l)
        local records
        records=$(gzip -dfc "$file" | grep -c "^--$")
        python3 -c "
file_path = \"$file\"
total_bytes = $total_bytes
lines = $lines
records = $records
avg_lines = lines / records if records > 0 else 0
print(f\"Analyzing '{file_path}'...\")
print(f\"Total size: {total_bytes:,} bytes\")
print(f\"\\n--- Exact Counts ---\")
print(f\"Total lines:          {lines:,}\")
print(f\"Total records:        {records:,}\")
print(f\"Average lines/record: {avg_lines:.2f}\")
"
        return
    fi

    local sample_lines
    sample_lines=$(head -c "$sample_bytes" "$file" | (gzip -dfc 2>/dev/null || true) | wc -l)
    local sample_records
    sample_records=$(head -c "$sample_bytes" "$file" | (gzip -dfc 2>/dev/null || true) | grep -c "^--$")

    python3 -c "
file_path = \"$file\"
total_bytes = $total_bytes
sample_bytes = $sample_bytes
sample_lines = $sample_lines
sample_records = $sample_records

density_lines = sample_lines / sample_bytes
density_records = sample_records / sample_bytes

est_lines = total_bytes * density_lines
est_records = total_bytes * density_records
avg_lines_per_record = sample_lines / sample_records if sample_records > 0 else 0

print(f\"Analyzing '{file_path}'...\")
print(f\"Total size: {total_bytes:,} bytes\")
print(f\"Sampling the first {sample_bytes / (1024*1024):.0f} MiB...\")
print(f\"\\n--- Estimation Results ---\")
print(f\"Sample lines:          {sample_lines:,}\")
print(f\"Sample records:        {sample_records:,}\")
print(f\"Estimated total lines: {est_lines:,.0f}\")
print(f\"Estimated records:     {est_records:,.0f} (average {avg_lines_per_record:.2f} lines/record)\")
"
}

if [[ $TODB -eq 1 ]]; then
    INPUT_FILE=""
    for arg in "$@"; do
        if [[ -f "$arg" ]]; then
            INPUT_FILE="$arg"
            break
        fi
    done
    if [[ -z $INPUT_FILE ]]; then
        INPUT_FILE="$1"
    fi
    
    if [[ -z $INPUT_FILE || "$INPUT_FILE" == "-" ]]; then
        DB_FILE="output.parquet"
    else
        DIR=$(dirname "$INPUT_FILE")
        BASE=$(basename "$INPUT_FILE")
        NAME="${BASE%.*}"
        if [[ "$BASE" =~ \.rec\.gz$ ]]; then
            NAME="${BASE%.rec.gz}"
        elif [[ "$NAME" == *.rec ]]; then
            NAME="${NAME%.rec}"
        fi
        DB_FILE="$DIR/$NAME.parquet"
    fi
    python3 "$MYTMP/helper.py" to "$DB_FILE" "$@"
    exit 0
fi

if [ "$#" -eq 0 ]; then
    usage; exit 0
fi

if [[ $GETNREC -eq 1 || $ESTNREC -eq 1 || $KEYS -eq 1 || $TOCSV -eq 1 || $TODB -eq 1 || $TOREC != 0 || $SAMPLE -gt 0 ]]; then
    CMD=""
    ENDCMD=""
    FILENAMES="$@"
else
    CMD=$(echo "$1" | tr "\n" "$" | gawk -F"END" '{print $1}' | tr "$" "\n")
    ENDCMD=$(echo "$1" | tr "\n" "$" | gawk -F"END" '{print $2}' | tr "$" "\n")
    FILENAMES="${@:2}"
fi

# Detect if input is a Parquet database
IS_DB=0
if [[ -n $FILENAMES && -f "$FILENAMES" ]]; then
    if [[ "$FILENAMES" =~ \.parquet$ ]] || head -c 4 "$FILENAMES" 2>/dev/null | grep -q "PAR1"; then
        IS_DB=1
    fi
fi

# Smart field filtering
FILTER=""
if [[ $TOREC == 0 && $KEYS == 0 && $TOCSV == 0 ]]; then
    # Detect fields used in the script: rec["field"] or rec['field']
    USED_FIELDS=$(echo "$1" | grep -oP "rec\[\s*['\"]\K[^'\"]+(?=['\"]\s*\])" | sort -u || true)
    # If the script iterates over all fields or uses printrec, we can't filter
    if [[ -n $USED_FIELDS ]] && ! echo "$1" | grep -qE "printrec|field\s+in\s+rec"; then
        FILTER="^--$"
        for f in $USED_FIELDS; do
            FILTER="$FILTER|^$f="
        done
    fi
fi

AWK_BIN="gawk"

if [[ $GETNREC -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        python3 -c "import sys, duckdb; print(duckdb.execute(f\"SELECT count(*) FROM '{sys.argv[1]}'\").fetchone()[0])" "$FILENAMES"
    else
        getnrec $FILENAMES
    fi
    exit 0
fi

if [[ $ESTNREC -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        local count
        count=$(python3 -c "import sys, duckdb; print(duckdb.execute(f\"SELECT count(*) FROM '{sys.argv[1]}'\").fetchone()[0])" "$FILENAMES")
        echo "Database file: '$FILENAMES'"
        echo "Exact records: $count"
    else
        estnrec $FILENAMES
    fi
    exit 0
fi

if [[ $KEYS -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        python3 -c "import sys, duckdb; print('\n'.join([r[0] for r in duckdb.execute(f\"DESCRIBE SELECT * FROM '{sys.argv[1]}'\").fetchall() if r[0] != 'rowid']))" "$FILENAMES"
    else
        gawk -F"=" '{
            if (FNR==1){
                fnr=0
            }
            if ($0=="--"){
                fnr+=1
            }
            else{
                keys[$1]=1
            }
        }
        END{
            for (k in keys){
                print k
            }
        }' $FILENAMES
    fi
    exit 0
fi

if [[ $TOCSV -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        python3 "$MYTMP/helper.py" from "$FILENAMES" "NONE" "" | gawk '
        BEGIN {
            first = 1
        }
        {
            if (FNR == 1) {
                fnr = 0
            }
            if ($0 == "--") {
                if (first) {
                    # Print header
                    for (i = 1; i <= nkeys; i++) {
                        printf "%s", keys[i]
                        if (i < nkeys) printf ","
                    }
                    printf "\n"
                    first = 0
                }
                # Print values for current record
                for (i = 1; i <= nkeys; i++) {
                    key = keys[i]
                    printf "%s", rec[key]
                    if (i < nkeys) printf ","
                }
                printf "\n"
                delete rec
            } else {
                # Store key-value pairs
                idx = index($0, "=")
                key = substr($0, 1, idx - 1)
                value = substr($0, idx + 1)
                rec[key] = value
                # Add key to keys array if not already present
                found = 0
                for (i = 1; i <= nkeys; i++) {
                    if (keys[i] == key) {
                        found = 1
                        break
                    }
                }
                if (!found) {
                    keys[++nkeys] = key
                }
            }
        }
        '
    else
        gawk '
        BEGIN {
            first = 1
        }
        {
            if (FNR == 1) {
                fnr = 0
            }
            if ($0 == "--") {
                if (first) {
                    # Print header
                    for (i = 1; i <= nkeys; i++) {
                        printf "%s", keys[i]
                        if (i < nkeys) printf ","
                    }
                    printf "\n"
                    first = 0
                }
                # Print values for current record
                for (i = 1; i <= nkeys; i++) {
                    key = keys[i]
                    printf "%s", rec[key]
                    if (i < nkeys) printf ","
                }
                printf "\n"
                delete rec
            } else {
                # Store key-value pairs
                idx = index($0, "=")
                key = substr($0, 1, idx - 1)
                value = substr($0, idx + 1)
                rec[key] = value
                # Add key to keys array if not already present
                found = 0
                for (i = 1; i <= nkeys; i++) {
                    if (keys[i] == key) {
                        found = 1
                        break
                    }
                }
                if (!found) {
                    keys[++nkeys] = key
                }
            }
        }
        ' "$FILENAMES"
    fi
    exit 0
fi

if [[ $TOREC != 0 ]]; then
    if [[ -z $TOREC ]]; then
        echo "Error: --torec requires a separator argument." >&2
        exit 1
    fi
    if [[ $TOREC == " " ]]; then
        gawk -v FPAT="[^[:space:]]+|(\"([^\"]|\"\")*\")" '{
            if (NR==1){
                for (i=1; i<=NF; i++){
                    # gsub(/ /, "_", $i)
                    keys[i]=$i
                }
            }
            else{
                for (i=1; i<=NF; i++){
                    if (keys[i] != ""){
                        print keys[i]"="$i
                    }
                }
                print "--"
            }
        }' "$FILENAMES"
    else
        gawk -v FPAT="[^$TOREC]*|(\"([^\"]|\"\")*\")" '{
            if (NR==1){
                for (i=1; i<=NF; i++){
                    # gsub(/ /, "_", $i)
                    keys[i]=$i
                }
            }
            else{
                for (i=1; i<=NF; i++){
                    if (keys[i] != ""){
                        print keys[i]"="$i
                    }
                }
                print "--"
            }
        }' "$FILENAMES"
    fi
    exit 0
fi

if [[ $SAMPLE -gt 0 ]]; then
    # We change the CMD to store the record in a reservoir instead of printing immediately.
    V="SAMPLE=$SAMPLE"
    # We wrap the user's command to run only at the END on the sampled records
    CMD='{
        # Reservoir Sampling Logic
        if (nr < SAMPLE) {
            # Fill the reservoir initially
            for (key in rec) reservoir[nr, key] = rec[key]
            res_keys[nr] = 1
        } else {
            # Replace with decreasing probability
            r = int((nr + 1) * rand())
            if (r < SAMPLE) {
                # Clear old record at index r and replace
                for (key in rec) {
                    # We use a 2D array simulation to store multiple records
                    reservoir[r, key] = rec[key]
                }
            }
        }
    }'
    # At the END, we loop through the reservoir and run the user's code
    ENDCMD='
        for (i=0; i < SAMPLE; i++) {
            # Restore the "rec" array for the current sampled record
            delete rec
            for (combined_key in reservoir) {
                split(combined_key, parts, SUBSEP)
                if (parts[1] == i) {
                    rec[parts[2]] = reservoir[combined_key]
                }
            }
            printrec();
            print("--")
            # Simulate the NR/FNR for the sample and run user command
            nr = i + 1; fnr = i + 1;
        }
    '
fi

# Define the AWK script parts to avoid duplication
AWK_FUNCTIONS=$(cat << 'EOF'
function printrec(){
    for (field in rec){
        print field"="rec[field]
    }
}

function spearman(x, y, n) {
    # Create arrays for ranking
    delete rank_x
    delete rank_y
    delete sorted_x
    delete sorted_y
    
    # Copy arrays
    for (i = 1; i <= n; i++) {
        sorted_x[i] = x[i]
        sorted_y[i] = y[i]
    }
    
    # Sort arrays
    for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
            if (sorted_x[i] > sorted_x[j]) {
                temp = sorted_x[i]
                sorted_x[i] = sorted_x[j]
                sorted_x[j] = temp
            }
            if (sorted_y[i] > sorted_y[j]) {
                temp = sorted_y[i]
                sorted_y[i] = sorted_y[j]
                sorted_y[j] = temp
            }
        }
    }
    
    # Assign ranks
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            if (x[i] == sorted_x[j]) {
                rank_x[i] = j
                break
            }
        }
        for (j = 1; j <= n; j++) {
            if (y[i] == sorted_y[j]) {
                rank_y[i] = j
                break
            }
        }
    }
    
    # Calculate Spearman correlation
    sum_d2 = 0
    for (i = 1; i <= n; i++) {
        d = rank_x[i] - rank_y[i]
        sum_d2 += d * d
    }
    
    if (n > 1) {
        return 1 - (6 * sum_d2) / (n * (n * n - 1))
    } else {
        return 0
    }
}

function pearson(x, y, n) {
    # Calculate means
    mean_x = 0
    mean_y = 0
    for (i = 1; i <= n; i++) {
        mean_x += x[i]
        mean_y += y[i]
    }
    mean_x /= n
    mean_y /= n
    
    # Calculate Pearson correlation
    numerator = 0
    sum_sq_x = 0
    sum_sq_y = 0
    for (i = 1; i <= n; i++) {
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        numerator += dx * dy
        sum_sq_x += dx * dx
        sum_sq_y += dy * dy
    }
    
    if (sum_sq_x > 0 && sum_sq_y > 0) {
        return numerator / sqrt(sum_sq_x * sum_sq_y)
    } else {
        return 0
    }
}
EOF
)

AWK_MAIN_LOOP_BEGIN=$(cat << 'EOF'
BEGIN{
srand(seed)
nr=0
}
{
if (FNR==1){
    fnr=0
}
if ($0=="--"){
    nr+=1
    fnr+=1
EOF
)

AWK_MAIN_LOOP_END=$(cat << 'EOF'
    delete rec
}
else{
    rec[$1]=substr($0,length($1)+2)
}
}
END{
EOF
)

FULL_AWK_SCRIPT="${AWK_FUNCTIONS}
${AWK_MAIN_LOOP_BEGIN}
${CMD}
${AWK_MAIN_LOOP_END}
${ENDCMD}
}"

if [[ $IS_DB -eq 1 ]]; then
    USED_FIELDS_CSV=""
    if [[ -n $USED_FIELDS ]] && ! echo "$1" | grep -qE "printrec|field\s+in\s+rec"; then
        USED_FIELDS_CSV=$(echo "$USED_FIELDS" | paste -sd, -)
    fi
    
    WHERE_CLAUSE="NONE"
    if [[ -n $WHERE ]]; then
        WHERE_CLAUSE="$WHERE"
    fi
    
    python3 "$MYTMP/helper.py" from "$FILENAMES" "$WHERE_CLAUSE" "$USED_FIELDS_CSV" | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
else
    if [[ -n $FILTER ]]; then
        if [[ -z $FILENAMES ]]; then
            grep -E "$FILTER" | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
        else
            grep -E "$FILTER" $FILENAMES | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
        fi
    else
        $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT" $FILENAMES
    fi
fi
