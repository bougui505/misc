"""Acoustic sync test (1Hz click) and microphone auto-calibration for multiroom audio."""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

import audio_backend

SAMPLE_RATE = 48000
SYNC_CLICK_PATH = Path("/tmp/multiroom_sync_click.wav")

_sync_lock = threading.Lock()
_sync_stop = threading.Event()
_sync_thread: threading.Thread | None = None
_sync_proc: subprocess.Popen[bytes] | None = None


def _make_chirp(duration_s: float = 0.025, f0: float = 800.0, f1: float = 4000.0) -> np.ndarray:
    """Generate a Hann-windowed logarithmic sine sweep."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False)
    k = np.log(f1 / f0) / duration_s
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    sig = np.sin(phase) * np.hanning(n)
    return (0.85 * sig).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono 16-bit PCM WAV file."""
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray:
    """Read mono 16-bit PCM WAV file into float32 array."""
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def find_chirp_offset_ms(recorded: np.ndarray, chirp: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Find arrival offset (ms) of chirp in recorded signal via FFT cross-correlation."""
    n_fft = 1 << (len(recorded) + len(chirp) - 1).bit_length()
    rec_f = np.fft.rfft(recorded, n=n_fft)
    chr_f = np.fft.rfft(chirp, n=n_fft)
    corr = np.fft.irfft(rec_f * np.conj(chr_f), n=n_fft)
    peak_idx = int(np.argmax(np.abs(corr[: len(recorded)])))
    return peak_idx / sample_rate * 1000.0


def is_sync_test_running() -> bool:
    """Return True if the 1Hz sync click loop is active."""
    return _sync_thread is not None and _sync_thread.is_alive() and not _sync_stop.is_set()


def stop_sync_test() -> dict[str, Any]:
    """Stop the looping 1Hz sync click."""
    global _sync_proc, _sync_thread
    with _sync_lock:
        _sync_stop.set()
        if _sync_proc is not None and _sync_proc.poll() is None:
            _sync_proc.terminate()
            try:
                _sync_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _sync_proc.kill()
        _sync_proc = None
        _sync_thread = None
    return {"ok": True, "running": False}


def start_sync_test(sink_name: str = audio_backend.MULTIROOM_SINK_NAME) -> dict[str, Any]:
    """Start a looping 1Hz sync click on the given sink."""
    global _sync_thread
    stop_sync_test()

    chirp = _make_chirp(duration_s=0.015, f0=1000.0, f1=3500.0)
    one_sec = np.zeros(SAMPLE_RATE, dtype=np.float32)
    one_sec[: len(chirp)] = chirp
    _write_wav(SYNC_CLICK_PATH, one_sec)

    _sync_stop.clear()

    def _loop() -> None:
        global _sync_proc
        while not _sync_stop.is_set():
            proc = subprocess.Popen(
                ["paplay", f"--device={sink_name}", str(SYNC_CLICK_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with _sync_lock:
                _sync_proc = proc
            proc.wait()

    _sync_thread = threading.Thread(target=_loop, daemon=True)
    _sync_thread.start()
    return {"ok": True, "running": True, "sink": sink_name}


def _measure_sink_delay_ms(sink_name: str, chirp_wav: Path, chirp: np.ndarray) -> float:
    """Play chirp on sink_name while recording from default mic, return arrival time in ms."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        rec_path = Path(tf.name)
    try:
        rec_proc = subprocess.Popen(
            [
                "parecord",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--format=s16le",
                "--file-format=wav",
                str(rec_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.25)
        subprocess.run(
            ["paplay", f"--device={sink_name}", str(chirp_wav)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        time.sleep(0.45)
        rec_proc.terminate()
        rec_proc.wait(timeout=3)
        recorded = _read_wav(rec_path)
        return find_chirp_offset_ms(recorded, chirp)
    finally:
        rec_path.unlink(missing_ok=True)


def auto_calibrate(sink_a: str, sink_b: str, apply: bool = True) -> dict[str, Any]:
    """Measure acoustic delay difference between sink_a and sink_b and optionally compensate."""
    sinks_map = {s["name"]: s for s in audio_backend.list_sinks()}
    if sink_a not in sinks_map or sink_b not in sinks_map:
        raise ValueError("Both selected sinks must be connected and active.")

    for name in (sink_a, sink_b):
        info = sinks_map[name]
        if info.get("card_name") and info.get("active_port"):
            audio_backend.set_latency_offset(info["card_name"], info["active_port"], 0)

    chirp = _make_chirp(duration_s=0.025)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        chirp_wav = Path(tf.name)
    try:
        _write_wav(chirp_wav, chirp)
        t_a_ms = _measure_sink_delay_ms(sink_a, chirp_wav, chirp)
        time.sleep(0.2)
        t_b_ms = _measure_sink_delay_ms(sink_b, chirp_wav, chirp)
    finally:
        chirp_wav.unlink(missing_ok=True)

    diff_ms = round(t_b_ms - t_a_ms)
    applied_to: str | None = None
    if apply and diff_ms != 0:
        # Delay the faster sink so it matches the slower one
        faster = sink_a if diff_ms > 0 else sink_b
        info = sinks_map[faster]
        if info.get("card_name") and info.get("active_port"):
            audio_backend.set_latency_offset(info["card_name"], info["active_port"], abs(diff_ms))
            applied_to = faster

    return {
        "ok": True,
        "sink_a": sink_a,
        "sink_b": sink_b,
        "delay_a_ms": round(t_a_ms, 1),
        "delay_b_ms": round(t_b_ms, 1),
        "diff_ms": diff_ms,
        "applied_to": applied_to,
        "offset_ms": abs(diff_ms),
    }
