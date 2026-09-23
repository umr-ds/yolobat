"""Detect bat calls in a recording, a directory of recordings, or live from a microphone.

The audio is analyzed one second at a time and detections are printed as they
are found; only the source of the chunks differs between the three modes.

Microphone mode needs alsa-utils and reads at the model's sample rate, letting
ALSA convert from the hardware's own rate (250 kHz for an UltraMic 250K). That
rate is checked at startup, because plug would upsample an ordinary 48 kHz
microphone just as happily, and nothing in the result would be a bat.

Usage
-----
    python detect.py -m model.onnx -i recording.wav
    python detect.py -m model.onnx -i recordings/ --recursive
    python detect.py -m model.onnx -i recording.wav --save-plots -o out/
    python detect.py -m model.onnx --mic
    python detect.py --mic --list-devices
"""
# SPDX-License-Identifier: MIT

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from yolobat import (DETECTION_THRESHOLD, compute_spec_size, load_audio, load_model,
                     row_to_freq, run_once)

QUEUE_SIZE = 8  # seconds of audio buffered before the oldest chunk is dropped
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


class Analyzer:
    """One ONNX session plus the settings recorded in the model it came from."""

    def __init__(self, model_path, threshold=DETECTION_THRESHOLD, hop_length=None):
        self.session, self.input_name, settings = load_model(model_path)
        self.sr = settings["sample_rate"]
        self.n_fft = settings["n_fft"]
        self.hop_length = hop_length or settings["hop_length"]
        self.class_names = settings["class_names"]
        self.threshold = threshold
        self.height, self.width = compute_spec_size(self.n_fft, self.hop_length, sr=self.sr)

    def run(self, clip):
        _, detections, spec, original_width = run_once(
            self.session, self.input_name, clip, self.height, self.width,
            self.n_fft, self.hop_length, self.threshold, self.sr,
            class_names=self.class_names,
        )
        return detections, spec, original_width


def print_chunk(label, detections, extra=""):
    """Prints a second only when it has something to say."""
    if not detections and not extra:
        return
    print(f"{label}: {len(detections)} detection(s){extra}", flush=True)
    for d in detections:
        print(f"    t=[{d['start_time']:.3f}, {d['end_time']:.3f}]s  "
              f"f=[{d['low_freq']:.0f}, {d['high_freq']:.0f}]Hz  "
              f"score={d['score']:.2f}  {d['class_name']}")


def save_detection_plot(path, spec_image, detections, second_index, duration, min_freq, max_freq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spec_image, aspect="auto", origin="upper",
              extent=[0, duration, min_freq, max_freq], cmap="magma", vmin=0, vmax=1)
    for d in detections:
        ax.add_patch(Rectangle(
            (d["start_time"], d["low_freq"]),
            d["end_time"] - d["start_time"], d["high_freq"] - d["low_freq"],
            linewidth=1.5, edgecolor="lime", facecolor="none",
        ))
        ax.text(d["start_time"], d["high_freq"], f"{d['class_name']} {d['score']:.2f}",
                color="lime", fontsize=7, va="bottom")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"second {second_index}: {len(detections)} detection(s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def find_audio_files(root, recursive):
    entries = Path(root).rglob("*") if recursive else Path(root).iterdir()
    return sorted(str(p) for p in entries
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def analyze_file(analyzer, path, args, label_prefix="", header=None):
    """Every whole second of one file; an unreadable file is skipped, not fatal."""
    try:
        wav = load_audio(path, analyzer.sr)
    except Exception as exc:
        print(f"{path}: could not read ({exc}) -- skipped", file=sys.stderr, flush=True)
        return 0

    num_chunks = len(wav) // analyzer.sr
    if num_chunks < 1:
        print(f"{path}: shorter than 1 second at {analyzer.sr} Hz -- skipped",
              file=sys.stderr, flush=True)
        return 0
    if args.max_seconds is not None:
        num_chunks = min(num_chunks, args.max_seconds)

    found = 0
    for i in range(num_chunks):
        detections, spec, original_width = analyzer.run(wav[i * analyzer.sr:(i + 1) * analyzer.sr])
        found += len(detections)
        if detections and header:
            print(header, flush=True)   # named only once, and only if it has calls in it
            header = None
        print_chunk(f"{label_prefix}second {i}", detections)
        if args.save_plots:
            stem = Path(path).stem
            save_detection_plot(
                os.path.join(args.out_dir, f"{stem}_second_{i:03d}.png"),
                spec[0, 2, :, :original_width], detections, i,
                original_width * analyzer.hop_length / analyzer.sr,
                row_to_freq(analyzer.height - 1, analyzer.n_fft, analyzer.sr, analyzer.height),
                row_to_freq(0, analyzer.n_fft, analyzer.sr, analyzer.height),
            )
    return found


class Mic:
    """Live capture through an `arecord` subprocess, so the Pi needs only alsa-utils.

    Records S16_LE, what USB ultrasonic microphones deliver, through the plug
    layer, which fixes up the channel count.
    """

    RATE_RE = re.compile(r"^RATE:\s*\[?(\d+)\s*(\d+)?", re.MULTILINE)

    def __init__(self, device, rate):
        self.bytes_per_chunk = rate * 2  # S16_LE, mono
        cmd = ["arecord", "-q", "-D", self.plug_device(device), "-f", "S16_LE",
               "-c", "1", "-r", str(int(rate)), "-t", "raw"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self):
        buf = bytearray()
        while len(buf) < self.bytes_per_chunk:  # a pipe read can come up short
            part = self.proc.stdout.read(self.bytes_per_chunk - len(buf))
            if not part:
                err = self.proc.stderr.read().decode(errors="replace").strip()
                raise RuntimeError(f"arecord stopped delivering audio{': ' + err if err else ''}")
            buf.extend(part)
        return np.frombuffer(bytes(buf), dtype="<i2").astype(np.float32) / 32768.0

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()

    @staticmethod
    def plug_device(device):
        """Always record through the plug layer, so ALSA converts format and
        channel count for us: a raw hw: device often offers only stereo S16_LE."""
        device = device or "default"
        return "plug" + device if device.startswith("hw:") else device

    @staticmethod
    def hw_device(device):
        """The raw hw: device behind --device, or None when it is not one."""
        device = device or "default"
        if device.startswith("plughw:"):
            return device[len("plug"):]
        return device if device.startswith("hw:") else None

    @classmethod
    def native_rate(cls, device):
        """The highest rate the hardware itself does, from `arecord --dump-hw-params`,
        or None when the device is not a plain hw:/plughw: one. Used only to tell
        an ultrasonic microphone from an ordinary one."""
        hw = cls.hw_device(device)
        if hw is None:
            return None
        try:
            proc = subprocess.run(
                ["arecord", "-D", hw, "--dump-hw-params", "-t", "raw", "-s", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5,
            )
        except Exception:
            return None
        match = cls.RATE_RE.search(proc.stderr.decode(errors="replace"))
        return int(match.group(2) or match.group(1)) if match else None

    @staticmethod
    def list_devices():
        print("Input devices (arecord -l) -- use e.g. --device plughw:<card>,0:", flush=True)
        subprocess.run(["arecord", "-l"])


def capture_loop(mic, chunk_queue, stop_event, drops):
    """Producer thread: keeps the device buffer drained, dropping the oldest pending
    chunk when analysis falls behind. Queue items are (seq, clip), an Exception if
    the stream died, or None when the thread is done."""
    seq = 0
    while not stop_event.is_set():
        try:
            item = (seq, mic.read())
        except Exception as exc:
            item = exc
        while True:
            try:
                chunk_queue.put_nowait(item)
                break
            except queue.Full:
                chunk_queue.get_nowait()  # drop the oldest to make room
                drops[0] += 1
        if isinstance(item, Exception):
            return
        seq += 1
    chunk_queue.put(None)


def run_microphone(analyzer, args):
    native = Mic.native_rate(args.device)
    warning = ("  WARNING: that is below 192 kHz, so most bat calls are not in the signal."
               if native and native < 192000 else "")
    print(f"Capturing from {Mic.plug_device(args.device)} at {analyzer.sr} Hz "
          f"(microphone runs at {f'{native} Hz' if native else 'an unverifiable rate'}"
          f"{'; ALSA converts' if native and native != analyzer.sr else ''})." + warning)

    mic = Mic(args.device, analyzer.sr)
    print("Listening... press Ctrl+C to stop.\n", flush=True)

    chunk_queue = queue.Queue(maxsize=QUEUE_SIZE)
    stop_event = threading.Event()
    drops = [0]  # list, so the producer thread can bump it in place
    thread = threading.Thread(target=capture_loop,
                              args=(mic, chunk_queue, stop_event, drops), daemon=True)
    thread.start()

    started = time.time()
    try:
        while True:
            if args.duration is not None and time.time() - started >= args.duration:
                break
            try:
                item = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            seq, clip = item
            detections, _, _ = analyzer.run(clip)
            extra = ""
            if chunk_queue.qsize():
                extra += f", backlog {chunk_queue.qsize()}"
            if drops[0]:
                extra += f", dropped {drops[0]}"
            print_chunk(f"second {seq}", detections, extra)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        stop_event.set()
        thread.join(timeout=3)  # let the in-flight read finish before closing under it
        mic.close()

    if drops[0]:
        print(f"Chunks dropped because analysis fell behind capture: {drops[0]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect bat calls in a recording, a directory of recordings, "
                    "or live microphone input.",
        epilog="Examples:\n"
               "  python detect.py -m model.onnx -i recording.wav\n"
               "  python detect.py -m model.onnx -i recordings/ --recursive\n"
               "  python detect.py -m model.onnx --mic\n"
               "  python detect.py --mic --list-devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--model", help="Path to the ONNX model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", help="Audio file, or a directory of audio files")
    source.add_argument("--mic", action="store_true", help="Analyze live microphone input")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                        help=f"Detection score threshold (default: {DETECTION_THRESHOLD})")
    parser.add_argument("--hop-length", type=int, default=None,
                        help="Override the STFT hop length recorded in the model")

    files = parser.add_argument_group("file / directory mode")
    files.add_argument("-r", "--recursive", action="store_true",
                       help="With a directory input, also descend into subdirectories")
    files.add_argument("-n", "--max-seconds", type=int, default=None,
                       help="Process at most this many seconds per file")
    files.add_argument("--save-plots", action="store_true",
                       help="Write an annotated spectrogram per second (needs matplotlib)")
    files.add_argument("-o", "--out-dir", default="detections",
                       help="Where --save-plots writes its PNGs")

    mic = parser.add_argument_group("microphone mode")
    mic.add_argument("--device", default=None,
                     help="ALSA capture device, e.g. plughw:3,0 (see --list-devices). "
                          "Default: the ALSA default device")
    mic.add_argument("--duration", type=float, default=None,
                     help="Stop after this many seconds (default: until Ctrl+C)")
    mic.add_argument("--list-devices", action="store_true",
                     help="List the available input devices and exit")

    args = parser.parse_args()
    if not (args.mic and args.list_devices) and args.model is None:
        parser.error("-m/--model is required")
    return args


def main():
    args = parse_args()

    # Listing devices must not require a model.
    if args.mic and args.list_devices:
        Mic.list_devices()
        return

    analyzer = Analyzer(args.model, threshold=args.threshold, hop_length=args.hop_length)
    print(f"Model {args.model}, {analyzer.sr} Hz, hop {analyzer.hop_length}, "
          f"spectrogram {analyzer.height}x{analyzer.width}, threshold {args.threshold}")

    if args.mic:
        try:
            run_microphone(analyzer, args)
        except RuntimeError as exc:
            raise SystemExit(f"microphone error: {exc}")
        return

    if args.save_plots:
        os.makedirs(args.out_dir, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    elif os.path.isdir(args.input):
        paths = find_audio_files(args.input, args.recursive)
        if not paths:
            raise SystemExit(f"no audio files found in {args.input}")
        print(f"Found {len(paths)} audio file(s) in {args.input}\n")
    else:
        raise SystemExit(f"no such file or directory: {args.input}")

    total = 0
    many = len(paths) > 1
    for path in paths:
        total += analyze_file(analyzer, path, args, label_prefix="  " if many else "",
                              header=f"--- {path}" if many else None)
    print(f"\n{total} detection(s) across {len(paths)} file(s)"
          if len(paths) > 1 else f"\n{total} detection(s)")
    if args.save_plots:
        print(f"Annotated spectrograms written to {args.out_dir}/")


if __name__ == "__main__":
    main()
