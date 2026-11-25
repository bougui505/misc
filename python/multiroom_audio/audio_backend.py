"""Zero-dependency PipeWire/PulseAudio and BlueZ backend for multiroom audio."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

PRESETS_PATH = Path(__file__).resolve().parent / "presets.json"
MULTIROOM_SINK_NAME = "multiroom_bt"
AUDIO_SINK_UUID = "0000110b"


def _run(cmd: list[str], timeout: int = 5) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the completed process."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _pactl_json(args: list[str]) -> Any:
    """Run pactl with -f json and parse the JSON output."""
    proc = _run(["pactl", "-f", "json", *args], timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"pactl {' '.join(args)} failed")
    return json.loads(proc.stdout or "[]")


def _parse_volume_percent(volume_obj: dict[str, Any]) -> int:
    """Compute average channel volume percentage from pactl JSON volume dict."""
    if not isinstance(volume_obj, dict) or not volume_obj:
        return 100
    percentages: list[int] = []
    for ch_data in volume_obj.values():
        if isinstance(ch_data, dict):
            vp = str(ch_data.get("value_percent", ""))
            m = re.search(r"(\d+)%", vp)
            if m:
                percentages.append(int(m.group(1)))
    return round(sum(percentages) / len(percentages)) if percentages else 100


def list_sinks() -> list[dict[str, Any]]:
    """Return all PulseAudio/PipeWire sinks enriched with card and latency info."""
    sinks_raw = _pactl_json(["list", "sinks"])
    cards_raw = _pactl_json(["list", "cards"])

    cards_by_obj_id: dict[str, dict[str, Any]] = {}
    cards_by_name: dict[str, dict[str, Any]] = {}
    for card in cards_raw:
        obj_id = str((card.get("properties") or {}).get("object.id", ""))
        if obj_id:
            cards_by_obj_id[obj_id] = card
        cname = card.get("name") or ""
        if cname:
            cards_by_name[cname] = card

    result: list[dict[str, Any]] = []
    for sink in sinks_raw:
        name = sink.get("name") or ""
        desc = sink.get("description") or ""
        if not desc or desc == "(null)":
            desc = name
        active_port = sink.get("active_port") or ""
        props = sink.get("properties") or {}
        card = cards_by_obj_id.get(str(props.get("device.id", ""))) or cards_by_name.get(
            str(props.get("device.name", ""))
        )
        card_name = ""
        latency_offset_ms = 0

        if card is not None:
            card_name = str(card["index"]) if card.get("index") is not None else (card.get("name") or "")
            ports = card.get("ports") or {}
            if isinstance(ports, dict) and active_port in ports:
                offset_raw = str(ports[active_port].get("latency_offset", "0") or "0")
                m_off = re.search(r"-?\d+", offset_raw)
                offset_us = int(m_off.group(0)) if m_off else 0
                latency_offset_ms = round(offset_us / 1000)

        driver = sink.get("driver") or ""
        is_combined = name == MULTIROOM_SINK_NAME or driver == "module-combine-sink"

        result.append(
            {
                "index": sink.get("index"),
                "name": name,
                "description": desc,
                "is_bluetooth": name.startswith("bluez_"),
                "is_combined": is_combined,
                "volume": _parse_volume_percent(sink.get("volume") or {}),
                "mute": bool(sink.get("mute", False)),
                "active_port": active_port,
                "card_name": card_name,
                "latency_offset_ms": latency_offset_ms,
            }
        )

    # Sort: combined first, then Bluetooth, then others
    result.sort(key=lambda s: (not s["is_combined"], not s["is_bluetooth"], s["description"].lower()))
    return result


def _run_bt_script(commands: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    """Run a sequence of bluetoothctl commands via stdin."""
    return subprocess.run(
        ["bluetoothctl"],
        input="\n".join(commands) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from bluetoothctl output."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def list_bluetooth_controllers() -> list[dict[str, Any]]:
    """Return all Bluetooth controllers known to bluetoothctl."""
    proc = _run(["bluetoothctl", "list"], timeout=5)
    if proc.returncode != 0:
        return []
    controllers: list[dict[str, Any]] = []
    for line in _strip_ansi(proc.stdout).splitlines():
        m = re.match(r"^Controller\s+([0-9A-Fa-f:]{17})\s+(.+?)(?:\s+\[default\])?$", line.strip())
        if m:
            controllers.append(
                {
                    "mac": m.group(1),
                    "name": m.group(2).strip(),
                    "is_default": "[default]" in line,
                }
            )
    return controllers


def list_bluetooth_devices() -> list[dict[str, Any]]:
    """Return paired Bluetooth devices deduplicated by MAC across all controllers."""
    controllers = list_bluetooth_controllers()
    if not controllers:
        controllers = [{"mac": "", "name": "default", "is_default": True}]

    by_mac: dict[str, dict[str, Any]] = {}
    for ctrl in controllers:
        if ctrl["mac"]:
            dev_proc = _run_bt_script([f"select {ctrl['mac']}", "devices", "quit"], timeout=5)
        else:
            dev_proc = _run(["bluetoothctl", "devices"], timeout=5)
        if dev_proc.returncode != 0:
            continue

        for line in _strip_ansi(dev_proc.stdout).splitlines():
            m = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.+)$", line.strip())
            if not m:
                continue
            mac, raw_name = m.group(1), m.group(2).strip()
            if ctrl["mac"]:
                info_proc = _run_bt_script([f"select {ctrl['mac']}", f"info {mac}", "quit"], timeout=5)
            else:
                info_proc = _run(["bluetoothctl", "info", mac], timeout=5)
            info_text = _strip_ansi(info_proc.stdout) if info_proc.returncode == 0 else ""

            name_match = re.search(r"^\s*Name:\s*(.+)$", info_text, re.MULTILINE)
            conn_match = re.search(r"^\s*Connected:\s*(yes|no)", info_text, re.MULTILINE | re.IGNORECASE)
            icon_match = re.search(r"^\s*Icon:\s*(.+)$", info_text, re.MULTILINE)

            icon = icon_match.group(1).strip() if icon_match else ""
            is_audio = (
                AUDIO_SINK_UUID in info_text.lower()
                or "audio" in icon.lower()
                or "headset" in icon.lower()
                or "headphone" in icon.lower()
                or "speaker" in icon.lower()
            )
            connected = (conn_match.group(1).lower() == "yes") if conn_match else False
            name = name_match.group(1).strip() if name_match else raw_name

            if mac not in by_mac:
                by_mac[mac] = {
                    "mac": mac,
                    "name": name,
                    "connected": connected,
                    "icon": icon,
                    "is_audio": is_audio,
                    "controller_mac": ctrl["mac"],
                    "controller_name": ctrl["name"],
                    "paired_controllers": [ctrl["mac"]] if ctrl["mac"] else [],
                }
            else:
                entry = by_mac[mac]
                if ctrl["mac"] and ctrl["mac"] not in entry["paired_controllers"]:
                    entry["paired_controllers"].append(ctrl["mac"])
                if connected:
                    entry["connected"] = True
                    entry["controller_mac"] = ctrl["mac"]
                    entry["controller_name"] = ctrl["name"]
                entry["is_audio"] = entry["is_audio"] or is_audio

    devices = list(by_mac.values())
    devices.sort(
        key=lambda d: (not d["is_audio"], not d["connected"], d["controller_name"].lower(), d["name"].lower())
    )
    return devices


def _disconnect_other_controllers(mac: str, keep_controller_mac: str) -> None:
    """Disconnect a device from any controller other than keep_controller_mac."""
    for ctrl in list_bluetooth_controllers():
        if ctrl["mac"] and ctrl["mac"] != keep_controller_mac:
            _run_bt_script([f"select {ctrl['mac']}", f"disconnect {mac}", "quit"], timeout=5)


def connect_bt(mac: str, controller_mac: str = "") -> dict[str, Any]:
    """Connect to a Bluetooth device by MAC address, optionally selecting a controller."""
    if controller_mac:
        _disconnect_other_controllers(mac, controller_mac)
        proc = _run_bt_script([f"select {controller_mac}", f"connect {mac}", "quit"], timeout=15)
        output = _strip_ansi(proc.stdout + "\n" + proc.stderr).strip()
        ok = proc.returncode == 0 and "Failed" not in output and "not available" not in output
        return {"ok": ok, "output": output}
    proc = _run(["bluetoothctl", "connect", mac], timeout=15)
    output = _strip_ansi(proc.stdout + "\n" + proc.stderr).strip()
    return {"ok": proc.returncode == 0, "output": output}


def pair_and_connect_bt(mac: str, controller_mac: str) -> dict[str, Any]:
    """Scan, pair, trust, and connect a device on the specified controller."""
    _disconnect_other_controllers(mac, controller_mac)
    proc = subprocess.Popen(
        ["bluetoothctl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(f"select {controller_mac}\npower on\nagent NoInputNoOutput\ndefault-agent\nscan on\n")
    proc.stdin.flush()
    time.sleep(6)

    proc.stdin.write(f"scan off\npair {mac}\n")
    proc.stdin.flush()
    time.sleep(5)

    proc.stdin.write(f"trust {mac}\nconnect {mac}\n")
    proc.stdin.flush()
    time.sleep(4)

    stdout, stderr = proc.communicate(input="quit\n", timeout=10)
    output = _strip_ansi(stdout + "\n" + stderr).strip()

    info_proc = _run_bt_script([f"select {controller_mac}", f"info {mac}", "quit"], timeout=5)
    info_out = _strip_ansi(info_proc.stdout)
    ok = "Connected: yes" in info_out or "Connection successful" in output

    res: dict[str, Any] = {"ok": ok, "output": output}
    if not ok:
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        res["error"] = " | ".join(lines[-3:]) if lines else "Pairing/connection failed"
    return res


def disconnect_bt(mac: str, controller_mac: str = "") -> dict[str, Any]:
    """Disconnect a Bluetooth device by MAC address, optionally selecting a controller."""
    if controller_mac:
        proc = _run_bt_script([f"select {controller_mac}", f"disconnect {mac}", "quit"], timeout=15)
        output = _strip_ansi(proc.stdout + "\n" + proc.stderr).strip()
        ok = proc.returncode == 0 and "Failed" not in output and "not available" not in output
        return {"ok": ok, "output": output}
    proc = _run(["bluetoothctl", "disconnect", mac], timeout=15)
    output = _strip_ansi(proc.stdout + "\n" + proc.stderr).strip()
    return {"ok": proc.returncode == 0, "output": output}


def disable_multiroom() -> dict[str, Any]:
    """Unload any existing multiroom_bt module-combine-sink."""
    modules = _pactl_json(["list", "modules"])
    unloaded: list[int] = []
    for mod in modules:
        if mod.get("name") == "module-combine-sink":
            arg = str(mod.get("argument") or "")
            if f"sink_name={MULTIROOM_SINK_NAME}" in arg:
                idx = mod.get("index")
                if idx is not None:
                    _run(["pactl", "unload-module", str(idx)], timeout=5)
                    unloaded.append(int(idx))
    return {"ok": True, "unloaded": unloaded}


def enable_multiroom(sink_names: list[str], set_default: bool = True) -> dict[str, Any]:
    """Create a combined multiroom sink targeting the given sink names."""
    clean_sinks = [s.strip() for s in sink_names if s and s.strip() and s.strip() != MULTIROOM_SINK_NAME]
    if len(clean_sinks) < 2:
        raise ValueError("Select at least 2 audio sinks for multiroom.")

    disable_multiroom()

    slaves = ",".join(clean_sinks)
    proc = _run(
        [
            "pactl",
            "load-module",
            "module-combine-sink",
            f"sink_name={MULTIROOM_SINK_NAME}",
            f"slaves={slaves}",
            'sink_properties=device.description="Multiroom_Bluetooth"',
        ],
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "Failed to load module-combine-sink")

    if set_default:
        _run(["pactl", "set-default-sink", MULTIROOM_SINK_NAME], timeout=5)
        try:
            sink_inputs = _pactl_json(["list", "sink-inputs"])
            for si in sink_inputs:
                si_idx = si.get("index")
                if si_idx is not None:
                    _run(["pactl", "move-sink-input", str(si_idx), MULTIROOM_SINK_NAME], timeout=5)
        except Exception:
            pass

    return {"ok": True, "module_index": proc.stdout.strip(), "slaves": clean_sinks}


def hard_resync(sink_names: list[str]) -> dict[str, Any]:
    """Suspend sinks to flush speaker A2DP buffers, then recreate multiroom_bt."""
    clean_sinks = [s.strip() for s in sink_names if s and s.strip() and s.strip() != MULTIROOM_SINK_NAME]
    if len(clean_sinks) < 2:
        raise ValueError("Select at least 2 audio sinks to resync.")

    disable_multiroom()
    for sink in clean_sinks:
        _run(["pactl", "suspend-sink", sink, "1"], timeout=5)
    time.sleep(1.5)
    for sink in clean_sinks:
        _run(["pactl", "suspend-sink", sink, "0"], timeout=5)
    time.sleep(0.5)

    res = enable_multiroom(clean_sinks, set_default=True)
    res["resynced"] = clean_sinks
    return res


def set_volume(sink_name: str, volume: int) -> dict[str, Any]:
    """Set sink volume percentage (clamped 0..150)."""
    vol = max(0, min(150, int(volume)))
    proc = _run(["pactl", "set-sink-volume", sink_name, f"{vol}%"], timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to set volume for {sink_name}")
    return {"ok": True, "sink": sink_name, "volume": vol}


def set_mute(sink_name: str, mute: bool) -> dict[str, Any]:
    """Mute or unmute a sink."""
    val = "1" if mute else "0"
    proc = _run(["pactl", "set-sink-mute", sink_name, val], timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to set mute for {sink_name}")
    return {"ok": True, "sink": sink_name, "mute": bool(mute)}


def set_latency_offset(card_name: str, port_name: str, offset_ms: int) -> dict[str, Any]:
    """Set port latency offset in milliseconds (clamped -500..1000)."""
    ms = max(-500, min(1000, int(offset_ms)))
    us = ms * 1000
    proc = _run(["pactl", "set-port-latency-offset", card_name, port_name, str(us)], timeout=5)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to set latency offset for {card_name}")
    return {"ok": True, "card": card_name, "port": port_name, "offset_ms": ms}


def load_presets() -> dict[str, Any]:
    """Load saved multiroom presets from presets.json."""
    if not PRESETS_PATH.exists():
        return {}
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_preset(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Save or overwrite a named preset."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty.")
    presets = load_presets()
    presets[clean_name] = data
    PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")
    return {"ok": True, "presets": presets}


def delete_preset(name: str) -> dict[str, Any]:
    """Delete a named preset."""
    presets = load_presets()
    presets.pop(name, None)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")
    return {"ok": True, "presets": presets}
