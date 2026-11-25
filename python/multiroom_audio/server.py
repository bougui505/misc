#!/usr/bin/env python3
"""Local HTTP server for the Bluetooth Multiroom Audio Web UI."""

from __future__ import annotations

import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import audio_backend
import sync_calibrate
from qr_terminal import render_qr_terminal

INDEX_HTML = Path(__file__).resolve().parent / "index.html"


def _get_lan_ip() -> str:
    """Best-effort local LAN IP discovery without external traffic."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class MultiroomHandler(BaseHTTPRequestHandler):
    """HTTP request handler for static UI and JSON API."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean
        pass

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            try:
                html = INDEX_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/api/state":
            try:
                sinks = audio_backend.list_sinks()
                bt_controllers = audio_backend.list_bluetooth_controllers()
                bt_devices = audio_backend.list_bluetooth_devices()
                multiroom_active = any(s["is_combined"] for s in sinks)
                presets = audio_backend.load_presets()
                self._send_json(
                    200,
                    {
                        "sinks": sinks,
                        "bluetooth_controllers": bt_controllers,
                        "bluetooth_devices": bt_devices,
                        "multiroom_active": multiroom_active,
                        "sync_test_running": sync_calibrate.is_sync_test_running(),
                        "presets": presets,
                    },
                )
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        try:
            if self.path == "/api/bt/connect":
                mac = str(body.get("mac", "")).strip()
                controller_mac = str(body.get("controller_mac", "")).strip()
                if not mac:
                    raise ValueError("Missing mac")
                res = audio_backend.connect_bt(mac, controller_mac)
                self._send_json(200 if res["ok"] else 500, res)
                return

            if self.path == "/api/bt/disconnect":
                mac = str(body.get("mac", "")).strip()
                controller_mac = str(body.get("controller_mac", "")).strip()
                if not mac:
                    raise ValueError("Missing mac")
                res = audio_backend.disconnect_bt(mac, controller_mac)
                self._send_json(200 if res["ok"] else 500, res)
                return

            if self.path == "/api/bt/pair":
                mac = str(body.get("mac", "")).strip()
                controller_mac = str(body.get("controller_mac", "")).strip()
                if not mac or not controller_mac:
                    raise ValueError("Missing mac or controller_mac")
                res = audio_backend.pair_and_connect_bt(mac, controller_mac)
                self._send_json(200 if res["ok"] else 500, res)
                return

            if self.path == "/api/multiroom/enable":
                sinks = body.get("sinks") or []
                set_default = bool(body.get("set_default", True))
                res = audio_backend.enable_multiroom(sinks, set_default=set_default)
                self._send_json(200, res)
                return

            if self.path == "/api/multiroom/disable":
                res = audio_backend.disable_multiroom()
                self._send_json(200, res)
                return

            if self.path == "/api/multiroom/resync":
                sinks = body.get("sinks") or []
                res = audio_backend.hard_resync(sinks)
                self._send_json(200, res)
                return

            if self.path == "/api/sink/volume":
                sink = str(body.get("sink", "")).strip()
                volume = int(body.get("volume", 100))
                if not sink:
                    raise ValueError("Missing sink")
                res = audio_backend.set_volume(sink, volume)
                self._send_json(200, res)
                return

            if self.path == "/api/sink/mute":
                sink = str(body.get("sink", "")).strip()
                mute = bool(body.get("mute", False))
                if not sink:
                    raise ValueError("Missing sink")
                res = audio_backend.set_mute(sink, mute)
                self._send_json(200, res)
                return

            if self.path == "/api/sink/latency":
                card = str(body.get("card", "")).strip()
                port = str(body.get("port", "")).strip()
                offset_ms = int(body.get("offset_ms", 0))
                if not card or not port:
                    raise ValueError("Missing card or port")
                res = audio_backend.set_latency_offset(card, port, offset_ms)
                self._send_json(200, res)
                return

            if self.path == "/api/presets/save":
                name = str(body.get("name", "")).strip()
                preset_data = {
                    "sinks": body.get("sinks") or [],
                    "volumes": body.get("volumes") or {},
                    "offsets": body.get("offsets") or {},
                }
                res = audio_backend.save_preset(name, preset_data)
                self._send_json(200, res)
                return

            if self.path == "/api/presets/apply":
                name = str(body.get("name", "")).strip()
                presets = audio_backend.load_presets()
                if name not in presets:
                    raise ValueError(f"Preset '{name}' not found")
                preset = presets[name]
                sinks_list = preset.get("sinks") or []
                res = audio_backend.enable_multiroom(sinks_list, set_default=True)

                current_sinks = {s["name"]: s for s in audio_backend.list_sinks()}
                for s_name, vol in (preset.get("volumes") or {}).items():
                    if s_name in current_sinks:
                        audio_backend.set_volume(s_name, int(vol))
                for s_name, off_ms in (preset.get("offsets") or {}).items():
                    s_info = current_sinks.get(s_name)
                    if s_info and s_info.get("card_name") and s_info.get("active_port"):
                        audio_backend.set_latency_offset(
                            s_info["card_name"], s_info["active_port"], int(off_ms)
                        )
                self._send_json(200, {"ok": True, "applied": name, "multiroom": res})
                return

            if self.path == "/api/presets/delete":
                name = str(body.get("name", "")).strip()
                res = audio_backend.delete_preset(name)
                self._send_json(200, res)
                return

            if self.path == "/api/sync/test/start":
                sink = str(body.get("sink", audio_backend.MULTIROOM_SINK_NAME)).strip()
                res = sync_calibrate.start_sync_test(sink)
                self._send_json(200, res)
                return

            if self.path == "/api/sync/test/stop":
                res = sync_calibrate.stop_sync_test()
                self._send_json(200, res)
                return

            if self.path == "/api/sync/calibrate":
                sinks = body.get("sinks") or []
                if len(sinks) < 2:
                    raise ValueError("Select at least 2 sinks to calibrate.")
                apply = bool(body.get("apply", True))
                res = sync_calibrate.auto_calibrate(str(sinks[0]), str(sinks[1]), apply=apply)
                self._send_json(200, res)
                return

            self._send_json(404, {"error": "Not found"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Bluetooth Multiroom Audio Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MultiroomHandler)
    lan_ip = _get_lan_ip()
    net_url = f"http://{lan_ip}:{args.port}"
    print(f"Multiroom Audio Web UI running:")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  Network: {net_url}")
    print(f"\nScan to open on your phone:\n{render_qr_terminal(net_url)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
