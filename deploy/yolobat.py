"""YOLObat inference pipeline."""
# SPDX-License-Identifier: MIT

import ast
import math
import numpy as np
import scipy.fft
import soundfile
import soxr
import onnxruntime as ort
import time

DETECTION_THRESHOLD = 0.25

def load_model(path, providers=("CPUExecutionProvider",)):
    """Open a model and read its settings. Returns (session, input name, settings)."""
    session = ort.InferenceSession(path, providers=list(providers))
    meta = session.get_modelmeta().custom_metadata_map

    missing = [k for k in ("sample_rate", "n_fft", "hop_length", "names") if k not in meta]
    if missing:
        raise ValueError(
            f"{path} is missing the metadata {', '.join(missing)}. Export it with this "
            f"project's export.py, which records the settings the model was trained with."
        )

    names = ast.literal_eval(meta["names"])
    return session, session.get_inputs()[0].name, {
        "sample_rate": int(meta["sample_rate"]),
        "n_fft": int(meta["n_fft"]),
        "hop_length": int(meta["hop_length"]),
        "class_names": [names[i] for i in sorted(names)],
    }


def load_audio(path, target_sr):
    """Mono float32 audio at target_sr, resampled with soxr as librosa.load does."""
    wav, file_sr = soundfile.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != target_sr:
        wav = soxr.resample(wav, file_sr, target_sr, quality="HQ")
    return np.ascontiguousarray(wav, dtype=np.float32)


def min_max_normalization(spec):
    return (spec - np.min(spec)) / (np.max(spec) - np.min(spec))


def amplitude_to_db(magnitude, amin=1e-5, top_db=80.0):
    """Custom version of librosa.amplitude_to_db, fixed to ref=np.max; use librosa otherwise."""
    ref_value = magnitude.max()
    log_spec = 20.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 20.0 * np.log10(max(amin, ref_value))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def compute_spec_size(n_fft, hop_length, sr=256000, analyze_length=1.0, low_bins=0, high_bins=0):
    """(H, W) of the spectrogram image, as YoloDataset derives it in training."""
    total_bins = n_fft / 2 + 1
    n_frames = math.ceil((sr / hop_length) * analyze_length)
    H = math.floor((total_bins - low_bins - high_bins) / 32) * 32
    W = math.floor(n_frames / 32) * 32
    return H, W


def compute_stft_magnitude(wav, n_fft, hop_length):
    """Faster librosa.stft(window='hann', center=True, pad_mode='reflect'), via scipy.fft."""
    n = np.arange(n_fft)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)).astype(np.float32)  # periodic Hann

    padded = np.pad(wav, n_fft // 2, mode="reflect").astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    windowed = frames * window

    spec = scipy.fft.rfft(windowed, axis=-1, workers=-1)
    return np.abs(spec).T.astype(np.float32)  # (n_freq_bins, n_frames)


def create_spectrum(wav, target_height, target_width, n_fft, hop_length, timings=None):
    """Model input for one second: dB spectrogram and two frequency channels, zero-padded.

    timings, if given, collects the duration of each step.
    """
    t0 = time.time()
    magnitude = compute_stft_magnitude(wav, n_fft, hop_length)
    t1 = time.time()

    db_magnitude = amplitude_to_db(magnitude)
    t2 = time.time()

    db_magnitude = min_max_normalization(db_magnitude) * 255
    spec_db = db_magnitude[::-1, :]  # high freq -> row 0, matches training's freq flip
    t3 = time.time()

    H, W = spec_db.shape

    # sinusoidal frequency encoding
    freq = np.linspace(0, 1, H, endpoint=False, dtype=np.float32).reshape(H, 1)
    sin_ch = (np.sin(2 * np.pi * freq) + 1.0) * 127.5
    cos_ch = (np.cos(2 * np.pi * freq) + 1.0) * 127.5
    t4 = time.time()

    # Format reverses the channels in training, so the model wants [cos, sin, spec]
    spec_width = min(W, target_width)
    padded_array = np.zeros((3, target_height, target_width), dtype=np.uint8)
    padded_array[0, :, :spec_width] = cos_ch
    padded_array[1, :, :spec_width] = sin_ch
    padded_array[2, :, :spec_width] = spec_db[:, :spec_width]

    spec_db = np.expand_dims(padded_array.astype(np.float32), axis=0)
    t5 = time.time()

    if timings is not None:
        timings["stft"].append(t1 - t0)
        timings["amplitude_to_db"].append(t2 - t1)
        timings["normalize"].append(t3 - t2)
        timings["pos_enc"].append(t4 - t3)
        timings["stack_pad"].append(t5 - t4)

    return spec_db / 255, spec_width


def row_to_freq(row, n_fft, sr, height):
    bin_index = (height - 1) - row
    return bin_index * sr / n_fft


def col_to_time(col, hop_length, sr):
    return col * hop_length / sr


def decode_detections(raw, threshold, height, original_width, hop_length, sr, n_fft,
                      class_names):
    detections = []
    for x1, y1, x2, y2, conf, cls in raw:
        if conf < threshold or x1 >= original_width:
            continue
        x2 = min(x2, original_width)  # clip out the zero-padded silence region
        detections.append({
            "start_time": max(0.0, col_to_time(x1, hop_length, sr)),
            "end_time": max(0.0, col_to_time(x2, hop_length, sr)),
            "low_freq": max(0.0, row_to_freq(y2, n_fft, sr, height)),
            "high_freq": max(0.0, row_to_freq(y1, n_fft, sr, height)),
            "score": float(conf),
            "class_name": class_names[int(cls)],
        })
    return detections


def run_once(session, input_name, clip, height, width, n_fft, hop_length, threshold, sr, profile=None, class_names=None):
    """One second through the pipeline: (timings, detections, input, unpadded width)."""
    t0 = time.time()
    spec, original_width = create_spectrum(
        clip, height, width, n_fft=n_fft, hop_length=hop_length, timings=profile
    )
    t1 = time.time()
    raw = session.run(None, {input_name: spec})[0][0]
    t2 = time.time()
    detections = decode_detections(
        raw, threshold, height=height, original_width=original_width,
        hop_length=hop_length, sr=sr, n_fft=n_fft, class_names=class_names,
    )
    t3 = time.time()
    return (t1 - t0, t2 - t1, t3 - t2), detections, spec, original_width
