"""Measure the inference runtime, as reported in the paper.

Times each second of audio as pre-processing, the model forward pass and
post-processing.

Usage
-----
    python benchmark.py 200 -m model.onnx -w recording.wav
    python benchmark.py 200 -m model.onnx --random-seconds 30 --profile
"""
# SPDX-License-Identifier: MIT

import argparse

import numpy as np

from yolobat import DETECTION_THRESHOLD, compute_spec_size, load_audio, load_model, run_once


def report(name, times, n_runs):
    """Print mean, median, standard deviation and range of a set of durations."""
    times = np.array(times)
    print(f"{name} over {n_runs} runs: "
          f"mean {times.mean():.4f}s, median {np.median(times):.4f}s, "
          f"std {times.std():.4f}s (min {times.min():.4f}s, max {times.max():.4f}s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference on a bat-call spectrogram.")
    parser.add_argument("--hop-length", type=int, default=None,
                        help="Override the STFT hop length recorded in the model")
    parser.add_argument("-m", "--model", required=True, help="Path to the ONNX model")
    parser.add_argument("-w", "--wav", help="Path to the input WAV file (ignored if --random-seconds is given)")
    parser.add_argument("--random-seconds", type=float, default=None,
                         help="If given, ignore --wav and use this many seconds of random audio instead")
    parser.add_argument("-n", "--runs", type=int, default=None,
                         help="Number of 1-second chunks to process (default: all whole seconds in the file)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Untimed runs before timing starts; 2 is what the paper used")
    parser.add_argument("--threshold", type=float, default=DETECTION_THRESHOLD,
                        help=f"Detection score threshold (default: {DETECTION_THRESHOLD})")
    parser.add_argument("--profile", action="store_true",
                         help="Break preprocessing into its substeps (STFT, amplitude-to-db, "
                              "normalize, positional encoding, stack/pad) and report timing for each")
    args = parser.parse_args()
    if args.wav is None and args.random_seconds is None:
        parser.error("give either --wav or --random-seconds")
    return args


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf, linewidth=200)

    args = parse_args()

    ort_session, input_name, settings = load_model(args.model)
    sr = settings["sample_rate"]
    n_fft = settings["n_fft"]
    class_names = settings["class_names"]
    hop_length = args.hop_length or settings["hop_length"]

    if args.random_seconds is not None:
        wav = np.random.uniform(-1.0, 1.0, size=int(args.random_seconds * sr)).astype(np.float32)
    else:
        wav = load_audio(args.wav, sr)
    num_chunks = len(wav) // sr
    if num_chunks < 1:
        raise ValueError(f"input audio is shorter than 1 second at {sr} Hz; need at least one full chunk")
    n_runs = num_chunks if args.runs is None else min(args.runs, num_chunks)

    target_height, target_width = compute_spec_size(n_fft, hop_length, sr=sr)

    profile_timings = (
        {"stft": [], "amplitude_to_db": [], "normalize": [], "pos_enc": [], "stack_pad": []}
        if args.profile else None
    )

    warmup_clip = wav[:sr]
    for _ in range(args.warmup):
        run_once(ort_session, input_name, warmup_clip, target_height, target_width,
                 n_fft, hop_length, args.threshold, sr, profile_timings, class_names)

    if profile_timings is not None:
        for times in profile_timings.values():
            times.clear()

    pre_times, model_times, post_times, pipeline_times = [], [], [], []
    for i in range(n_runs):
        clip = wav[i * sr:(i + 1) * sr]
        (pre_t, model_t, post_t), detections, _, _ = run_once(
            ort_session, input_name, clip, target_height, target_width,
            n_fft, hop_length, args.threshold, sr, profile_timings, class_names)
        pre_times.append(pre_t)
        model_times.append(model_t)
        post_times.append(post_t)
        pipeline_times.append(pre_t + model_t + post_t)
        print(
            f"second {i}: pre {pre_t:.4f}s, model {model_t:.4f}s, "
            f"post {post_t:.4f}s, pipeline {pre_t + model_t + post_t:.4f}s, "
            f"{len(detections)} detection(s)",
            flush=True,
        )
        for d in detections:
            print(
                f"    t=[{d['start_time']:.3f}, {d['end_time']:.3f}]s  "
                f"f=[{d['low_freq']:.0f}, {d['high_freq']:.0f}]Hz  "
                f"score={d['score']:.2f}  {d['class_name']}"
            )

    report("Preprocessing", pre_times, n_runs)
    report("Model-only", model_times, n_runs)
    report("Postprocessing", post_times, n_runs)
    report("Whole pipeline (preprocessing + model + postprocessing)", pipeline_times, n_runs)

    if profile_timings is not None:
        print()
        for name, key in [
            ("Preprocessing/STFT", "stft"),
            ("Preprocessing/amplitude_to_db", "amplitude_to_db"),
            ("Preprocessing/normalize", "normalize"),
            ("Preprocessing/positional encoding", "pos_enc"),
            ("Preprocessing/stack+pad", "stack_pad"),
        ]:
            report(name, profile_timings[key], n_runs)
