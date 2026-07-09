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
import sqlite3
import os
import time

mode = sys.argv[1]

if mode == 'to':
    db_path = sys.argv[2]
    files = sys.argv[3:]
    
    class ProgressReporter:
        def __init__(self, files):
            self.files = files
            self.use_percentage = False
            self.total_bytes = 0
            self.record_count = 0
            self.last_update_time = 0
            self.current_fh = None
            self.processed_bytes_prev_files = 0
            
            if files and all(f != '-' for f in files):
                try:
                    self.total_bytes = sum(os.path.getsize(f) for f in files if os.path.exists(f))
                    if self.total_bytes > 0:
                        self.use_percentage = True
                except Exception:
                    self.use_percentage = False

        def set_fh(self, fh, prev_bytes):
            self.current_fh = fh
            self.processed_bytes_prev_files = prev_bytes

        def update(self, is_record=False):
            if is_record:
                self.record_count += 1
            
            now = time.time()
            if now - self.last_update_time >= 0.2:
                self.print_progress()
                self.last_update_time = now

        def print_progress(self):
            sys.stderr.write("\r\033[K")
            if self.use_percentage and self.current_fh:
                offset = 0
                fh = self.current_fh
                try:
                    if hasattr(fh, 'buffer'):
                        buf = fh.buffer
                        if hasattr(buf, 'fileobj'):
                            offset = buf.fileobj.tell()
                        elif hasattr(buf, 'tell'):
                            offset = buf.tell()
                    elif hasattr(fh, 'tell'):
                        offset = fh.tell()
                except Exception:
                    pass
                
                processed = self.processed_bytes_prev_files + offset
                pct = min(100.0, (processed / self.total_bytes) * 100)
                bar_width = 30
                filled = int(bar_width * pct / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                sys.stderr.write(f"Converting to DB: [{bar}] {pct:.1f}% ({self.record_count:,} records)")
            else:
                sys.stderr.write(f"Converting to DB: {self.record_count:,} records processed...")
            sys.stderr.flush()

        def print_final_progress(self):
            sys.stderr.write("\r\033[K")
            bar_width = 30
            bar = '█' * bar_width
            sys.stderr.write(f"Converting to DB: [{bar}] 100.0% ({self.record_count:,} records)")
            sys.stderr.flush()

        def finish(self):
            if self.use_percentage:
                self.print_final_progress()
            else:
                self.print_progress()
            sys.stderr.write("\nDone!\n")
            sys.stderr.flush()

    is_gz = db_path.endswith('.gz')
    if is_gz:
        temp_db_dir = os.environ.get('MYTMP', '/tmp')
        temp_db = os.path.join(temp_db_dir, 'temp_create.db')
        if os.path.exists(temp_db):
            try: os.remove(temp_db)
            except: pass
        active_db = temp_db
    else:
        active_db = db_path

    conn = sqlite3.connect(active_db)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS records (rowid INTEGER PRIMARY KEY AUTOINCREMENT)")
    conn.commit()

    known_cols = set()
    cursor.execute("PRAGMA table_info(records)")
    for row in cursor.fetchall():
        known_cols.add(row[1])

    current_record = {}
    cursor.execute("BEGIN TRANSACTION")
    count = 0
    
    reporter = ProgressReporter(files)
    prev_bytes = 0
    
    if not files:
        for line in sys.stdin:
            line_clean = line.rstrip('\r\n')
            if line_clean == "--":
                if current_record:
                    new_cols = set(current_record.keys()) - known_cols
                    for col in new_cols:
                        cursor.execute(f'ALTER TABLE records ADD COLUMN "{col}" TEXT')
                        known_cols.add(col)
                    cols = ', '.join(f'"{k}"' for k in current_record.keys())
                    placeholders = ', '.join('?' for _ in current_record)
                    cursor.execute(f'INSERT INTO records ({cols}) VALUES ({placeholders})', list(current_record.values()))
                    current_record.clear()
                    count += 1
                    reporter.update(is_record=True)
                    if count % 20000 == 0:
                        conn.commit()
                        cursor.execute("BEGIN TRANSACTION")
                else:
                    reporter.update(is_record=False)
            else:
                if '=' in line_clean:
                    parts = line_clean.split('=', 1)
                    current_record[parts[0]] = parts[1]
                reporter.update(is_record=False)
    else:
        for f in files:
            if f == '-':
                for line in sys.stdin:
                    line_clean = line.rstrip('\r\n')
                    if line_clean == "--":
                        if current_record:
                            new_cols = set(current_record.keys()) - known_cols
                            for col in new_cols:
                                cursor.execute(f'ALTER TABLE records ADD COLUMN "{col}" TEXT')
                                known_cols.add(col)
                            cols = ', '.join(f'"{k}"' for k in current_record.keys())
                            placeholders = ', '.join('?' for _ in current_record)
                            cursor.execute(f'INSERT INTO records ({cols}) VALUES ({placeholders})', list(current_record.values()))
                            current_record.clear()
                            count += 1
                            reporter.update(is_record=True)
                            if count % 20000 == 0:
                                conn.commit()
                                cursor.execute("BEGIN TRANSACTION")
                        else:
                            reporter.update(is_record=False)
                    else:
                        if '=' in line_clean:
                            parts = line_clean.split('=', 1)
                            current_record[parts[0]] = parts[1]
                        reporter.update(is_record=False)
            else:
                file_size = os.path.getsize(f) if os.path.exists(f) else 0
                if f.endswith('.gz'):
                    import gzip
                    fh = gzip.open(f, 'rt', encoding='utf-8', errors='ignore')
                else:
                    fh = open(f, 'r', encoding='utf-8', errors='ignore')
                    
                reporter.set_fh(fh, prev_bytes)
                
                try:
                    for line in fh:
                        line_clean = line.rstrip('\r\n')
                        if line_clean == "--":
                            if current_record:
                                new_cols = set(current_record.keys()) - known_cols
                                for col in new_cols:
                                    cursor.execute(f'ALTER TABLE records ADD COLUMN "{col}" TEXT')
                                    known_cols.add(col)
                                cols = ', '.join(f'"{k}"' for k in current_record.keys())
                                placeholders = ', '.join('?' for _ in current_record)
                                cursor.execute(f'INSERT INTO records ({cols}) VALUES ({placeholders})', list(current_record.values()))
                                current_record.clear()
                                count += 1
                                reporter.update(is_record=True)
                                if count % 20000 == 0:
                                    conn.commit()
                                    cursor.execute("BEGIN TRANSACTION")
                            else:
                                reporter.update(is_record=False)
                        else:
                            if '=' in line_clean:
                                parts = line_clean.split('=', 1)
                                current_record[parts[0]] = parts[1]
                            reporter.update(is_record=False)
                finally:
                    fh.close()
                prev_bytes += file_size
                
    if current_record:
        new_cols = set(current_record.keys()) - known_cols
        for col in new_cols:
            cursor.execute(f'ALTER TABLE records ADD COLUMN "{col}" TEXT')
            known_cols.add(col)
        cols = ', '.join(f'"{k}"' for k in current_record.keys())
        placeholders = ', '.join('?' for _ in current_record)
        cursor.execute(f'INSERT INTO records ({cols}) VALUES ({placeholders})', list(current_record.values()))
        reporter.update(is_record=True)
        
    conn.commit()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_rowid ON records(rowid)")
    conn.commit()
    conn.close()
    reporter.finish()
    
    if is_gz:
        import gzip
        import shutil
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass
        with open(temp_db, 'rb') as f_in:
            with gzip.open(db_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        try: os.remove(temp_db)
        except: pass

elif mode == 'from':
    db_path = sys.argv[2]
    where_clause = sys.argv[3] if sys.argv[3] != 'NONE' else None
    used_fields_str = sys.argv[4] if len(sys.argv) > 4 else ''
    
    cols_to_query = [f.strip() for f in used_fields_str.split(',') if f.strip()] if used_fields_str else None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records'")
    if not cursor.fetchone():
        sys.stderr.write("Error: 'records' table not found in database.\n")
        sys.exit(1)
        
    cursor.execute("PRAGMA table_info(records)")
    all_cols = [row[1] for row in cursor.fetchall() if row[1] != 'rowid']
    
    if cols_to_query:
        cols = [c for c in cols_to_query if c in all_cols]
    else:
        cols = all_cols
        
    if not cols:
        cursor.execute("SELECT count(*) FROM records")
        count = cursor.fetchone()[0]
        for _ in range(count):
            print("--")
        sys.exit(0)
        
    col_selectors = ', '.join(f'"{c}"' for c in cols)
    query = f'SELECT {col_selectors} FROM records'
    if where_clause:
        query += f' WHERE {where_clause}'
        
    cursor.execute(query)
    
    try:
        while True:
            rows = cursor.fetchmany(1000)
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
  --todb               Convert rec format to compressed SQLite database (uses input file basename)
  -w, --where CLAUSE   SQL WHERE clause to filter records when querying a database

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

  # Convert data to compressed SQLite database (automatically outputs to data.db.gz)
  recawk --todb data.rec

  # Query database seamlessly
  recawk '{print rec["tmscore"]}' data.db.gz

  # Query database with database-level filtering
  recawk -w "tmscore > 0.8" '{print rec["tmscore"]}' data.db.gz
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
        DB_FILE="output.db.gz"
    else
        DIR=$(dirname "$INPUT_FILE")
        BASE=$(basename "$INPUT_FILE")
        NAME="${BASE%.*}"
        if [[ "$BASE" =~ \.rec\.gz$ ]]; then
            NAME="${BASE%.rec.gz}"
        elif [[ "$NAME" == *.rec ]]; then
            NAME="${NAME%.rec}"
        fi
        DB_FILE="$DIR/$NAME.db.gz"
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

# Detect if input is a SQLite database
IS_DB=0
IS_DB_GZ=0
QUERY_DB_FILE=""
if [[ -n $FILENAMES && -f "$FILENAMES" ]]; then
    if [[ "$FILENAMES" =~ \.db\.gz$ || "$FILENAMES" =~ \.sqlite\.gz$ ]]; then
        IS_DB=1
        IS_DB_GZ=1
        QUERY_DB_FILE="$MYTMP/query.db"
        gzip -dc "$FILENAMES" > "$QUERY_DB_FILE"
    elif [[ "$FILENAMES" =~ \.(db|sqlite)$ ]] || head -c 15 "$FILENAMES" 2>/dev/null | grep -q "SQLite format 3"; then
        IS_DB=1
        QUERY_DB_FILE="$FILENAMES"
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
        sqlite3 "$QUERY_DB_FILE" "SELECT count(*) FROM records"
    else
        getnrec $FILENAMES
    fi
    exit 0
fi

if [[ $ESTNREC -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        local count
        count=$(sqlite3 "$QUERY_DB_FILE" "SELECT count(*) FROM records")
        echo "Database file: '$FILENAMES'"
        echo "Exact records: $count"
    else
        estnrec $FILENAMES
    fi
    exit 0
fi

if [[ $KEYS -eq 1 ]]; then
    if [[ $IS_DB -eq 1 ]]; then
        sqlite3 "$QUERY_DB_FILE" "PRAGMA table_info(records)" | cut -d'|' -f2 | grep -v "^rowid$"
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
        python3 "$MYTMP/helper.py" from "$QUERY_DB_FILE" "NONE" "" | gawk '
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
    
    python3 "$MYTMP/helper.py" from "$QUERY_DB_FILE" "$WHERE_CLAUSE" "$USED_FIELDS_CSV" | $AWK_BIN -v seed=$RANDOM -v SAMPLE=$SAMPLE -v $V -F"=" "$FULL_AWK_SCRIPT"
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
