#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import socket
import sqlite3
import threading
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import urllib.parse

# Configuration
PORT = 8080
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temperature.db')
SERIAL_PORT = '/dev/ttyACM0'

# Global state to share latest reading with API
latest_reading = {
    "temperature": None,
    "timestamp": None,
    "status": "disconnected",
    "error": "Starting up..."
}

def parse_temp(line):
    """
    Parses a line of text to extract a floating point temperature.
    Supports formats like:
      - '00:14:06 | Temp: 30.1°C | Tendance: STABLE'
      - 'Temp: 23.5 C'
      - 't = 22.4'
      - '23.4' (raw number)
    """
    line = line.strip()
    if not line:
        return None
        
    # Pattern 1: Look for temp/temperature key-value pairs (e.g. Temp: 23.5 or t = 22.4)
    # This avoids matching timestamps (e.g., "00:14:06") or tracebacks
    match = re.search(r'(?:temp|temperature|t)\s*[:=]\s*([-+]?\d*\.\d+|\d+)', line, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            if -40.0 <= val <= 80.0:
                return val
        except ValueError:
            pass
            
    # Pattern 2: Fallback to matching if the entire line is just a float (for raw output devices)
    match_raw = re.match(r'^[-+]?\d*\.\d+|[-+]?\d+$', line)
    if match_raw:
        try:
            val = float(match_raw.group())
            if -40.0 <= val <= 80.0:
                return val
        except ValueError:
            pass
            
    return None

def db_init():
    """Initializes the SQLite database structure."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            timestamp INTEGER NOT NULL,
            temperature REAL NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON readings(timestamp)
    ''')
    conn.commit()
    conn.close()

def serial_reader_thread():
    """Background thread to read data from the serial port and log to the database."""
    global latest_reading
    db_init()
    
    print(f"[*] Serial reader thread started. Monitoring: {SERIAL_PORT}")
    
    while True:
        try:
            if not os.path.exists(SERIAL_PORT):
                raise FileNotFoundError(f"Serial port {SERIAL_PORT} does not exist.")
            
            # Open serial port as a standard file
            with open(SERIAL_PORT, 'r', encoding='utf-8', errors='ignore') as f:
                print(f"[+] Successfully opened serial port {SERIAL_PORT}")
                latest_reading["status"] = "connected"
                latest_reading["error"] = None
                
                while True:
                    line = f.readline()
                    if not line:
                        # EOF indicates serial device disconnected
                        print("[-] Serial port returned EOF. Reconnecting in 2 seconds...")
                        time.sleep(2)
                        break
                    
                    print(f"[*] Raw serial line: {repr(line)}")
                    temp = parse_temp(line)
                    if temp is not None:
                        now = int(time.time())
                        
                        # Update global status
                        latest_reading["temperature"] = temp
                        latest_reading["timestamp"] = now
                        latest_reading["status"] = "connected"
                        latest_reading["error"] = None
                        
                        # Log to database
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO readings (timestamp, temperature) VALUES (?, ?)",
                                (now, temp)
                            )
                            conn.commit()
                            conn.close()
                        except Exception as db_err:
                            print(f"[!] Database insert error: {db_err}")
                            
        except Exception as e:
            # Update state with connection error
            latest_reading["status"] = "disconnected"
            latest_reading["error"] = str(e)
            print(f"[!] Serial reader error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def get_history(period):
    """
    Retrieves historical temperature readings with downsampling to keep charts fast.
    - 1h: raw values
    - 24h: 5-minute averages
    - 7d: 1-hour averages
    """
    now = int(time.time())
    
    if period == '1h':
        start_time = now - 3600
        interval = 10  # Group by 10s (practically raw)
    elif period == '24h':
        start_time = now - 86400
        interval = 300  # 5-minute groups
    elif period == '7d':
        start_time = now - 7 * 86400
        interval = 3600  # 1-hour groups
    else:
        start_time = now - 86400
        interval = 300

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT (timestamp / ?) * ? AS group_time, AVG(temperature)
            FROM readings
            WHERE timestamp >= ?
            GROUP BY group_time
            ORDER BY group_time ASC
        ''', (interval, interval, start_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            if row[1] is not None:
                history.append({
                    "timestamp": row[0],
                    "temperature": round(row[1], 2)
                })
        return history
    except Exception as e:
        print(f"[!] Error fetching history from database: {e}")
        return []

class ThermometerHTTPRequestHandler(BaseHTTPRequestHandler):
    """Handler for API endpoints and serving static frontend assets."""
    
    def log_message(self, format, *args):
        # Suppress standard logging to keep terminal output clean and readable
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # Route API requests
        if path == '/api/current':
            self.send_json(latest_reading)
        elif path == '/api/history':
            period = query.get('period', ['24h'])[0]
            self.send_json(get_history(period))
        # Route static files
        elif path in ('/', '/index.html'):
            self.serve_file('index.html', 'text/html')
        elif path == '/style.css':
            self.serve_file('style.css', 'text/css')
        elif path == '/app.js':
            self.serve_file('app.js', 'application/javascript')
        elif path == '/chart.js':
            self.serve_file('chart.js', 'application/javascript')
        else:
            self.send_error(404, "File not found")

    def send_json(self, data):
        try:
            content = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")

    def serve_file(self, filename, content_type):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filename} not found")
            return
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file {filename}: {e}")

def get_local_ips():
    """Gets all local IPv4 addresses of this machine."""
    ips = []
    # Primary outgoing route IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
        
    # Standard check_output for hostname -I (very reliable on Linux)
    try:
        output = subprocess.check_output(['hostname', '-I']).decode().strip()
        for ip in output.split():
            if ip not in ips and not ip.startswith('127.'):
                ips.append(ip)
    except Exception:
        pass
        
    return ips

def main():
    # Start the serial reader daemon thread
    reader_thread = threading.Thread(target=serial_reader_thread, daemon=True)
    reader_thread.start()
    
    # Configure the HTTP server
    server = ThreadingHTTPServer(('0.0.0.0', PORT), ThermometerHTTPRequestHandler)
    
    # Print welcome instructions with local IP addresses
    ips = get_local_ips()
    
    print("=" * 65)
    print("           📊 LOCAL TEMPERATURE MONITOR SERVER STARTED 📊")
    print("=" * 65)
    print(f"  • Local Dashboard:   http://localhost:{PORT}")
    print("  • Phone / Network Dashboard URLs:")
    if ips:
        for ip in ips:
            print(f"    👉  http://{ip}:{PORT}")
    else:
        print("    👉  http://<your-computer-ip-address>:" + str(PORT))
    print("-" * 65)
    print("  • Note: Ensure your phone and computer are on the same Wi-Fi network.")
    print("  • Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
        server.server_close()
        print("[+] Server stopped.")

if __name__ == '__main__':
    main()
