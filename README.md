# YOLObat

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.9.1](https://img.shields.io/badge/pytorch-2.9.1%2Bcu128-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics 8.4.21](https://img.shields.io/badge/ultralytics-8.4.21%20%2B%20patch-0b8f8c.svg)](ultralytics-yolobat.patch)
[![Paper: under review](https://img.shields.io/badge/paper-under%20review-lightgrey.svg)](#citation)

Bat species and behavior recognition in ultrasonic recordings, by applying
[YOLO26](https://docs.ultralytics.com) object detection to STFT spectrograms.
One inference pass detects 18 bat classes, three of which are species groups,
alongside feeding buzzes and social calls.

This repository holds the training pipeline for the paper below. Two ideas in it
are specific to bioacoustics:

- **Sinusoidal frequency positional encodings.** Mosaic augmentation moves image
  tiles around, which destroys the meaning of the frequency axis. Encoding
  frequency into two extra channels makes the absolute position recoverable, so
  mosaic becomes usable on spectrograms.
- **Taxonomy-aware label smoothing.** A fixed share of the target probability is
  redistributed only to species with similar calls, rather than uniformly across
  all classes. The share does not grow with group size, so the eight-member
  *Myotis* group is not smoothed more aggressively than a two-member one.

## Prerequisites

Training needs Docker with the NVIDIA container runtime and a GPU. Everything
else, including the patched Ultralytics, lives in the image.

Running a trained model needs neither: `deploy/` uses ONNX Runtime on the CPU,
which is what makes a Raspberry Pi viable. See [Deployment](#deployment).

## Quick start

**1. Build.** The image is built on top of Ultralytics 8.4.21.

```bash
docker compose build
```

**2. Set audio directory in .env.**

```bash
cp .env.example .env    # then set AUDIO_DIR
```

`AUDIO_DIR` is the only required variable, `RUNS_DIR`, `GPU_ID`, `SHM_SIZE` and
`TENSORBOARD_PORT` may be overridden.

**3. Train.**

```bash
docker compose run --rm yolobat python Main.py

# settings can be overridden with --set
docker compose run --rm yolobat python Main.py --set epochs=100 batch=64
```

Checkpoints, logs and plots appear under `runs/` on the host. TensorBoard is
served on `127.0.0.1:6006`.

Our trained checkpoints are available
[here](https://dshare.mathematik.uni-marburg.de/index.php/s/Z5bb4mDN96TAt3P):
the n, s and m models with five seeds each. Unpack them into `var/models/` and
point `model` at one to continue from it instead of training from scratch:

```bash
docker compose run --rm yolobat python Main.py --set model=var/models/yolobat-n/seed2.pt
```

The paper reports the mean over the five seeds. For a single model, seed 2 is the
best of the n and s runs and seed 4 of m.

`files/cfg.yaml` is the configuration the published models were trained with:
YOLO26n, 150 epochs, batch 16, MuSGD with a cosine schedule, mosaic with
positional encodings and taxonomy-aware smoothing at eps 0.05. The spectrogram
settings `sr`, `n_fft` and `hop_length` together fix the input size, 1280x384
covering 0-128 kHz.

## Data

Audio is read directly and turned into spectrograms on the fly, so no images are
stored. The annotations are a COCO file, `var/data_exports/yolobat_dataset.json`,
and the train / val split two lists beside it, `yolobat_train.txt` and
`yolobat_val.txt`, one recording per line. All paths are relative to `AUDIO_DIR`
and a `bbox` is `[time_start, frequency_start, duration, bandwidth]` in seconds
and hertz.

Easily confused species for taxonomy-aware smoothing can be set in `files/taxonomy.yaml`.

`files/noise/crickets.txt` takes the recordings the noise augmentation mixes into
training samples, one per line; they are then excluded from training itself. The
published list is empty, which turns the augmentation off.

## Export

To run the model on a Raspberry Pi, we recommend exporting it to ONNX:
`python export.py --config <run> --version yolobat --release 2026.1 --format onnx`

## Deployment

`deploy/detect.py` analyzes a recording, a directory of recordings, or a live
ultrasonic microphone, and `deploy/benchmark.py` measures the inference time
reported in the paper. Our trained main model is available in the ONNX format
[here](https://dshare.mathematik.uni-marburg.de/index.php/s/roMQ9MZfWqP262g),
with the settings it was trained with in its metadata.

```bash
pip install -r deploy/requirements.txt

python deploy/detect.py -m model.onnx -i recording.wav
python deploy/detect.py -m model.onnx -i recordings/ --recursive
python deploy/detect.py -m model.onnx --mic --device plughw:3,0  # --list-devices shows what is connected
python deploy/benchmark.py -m model.onnx --random-seconds 30 --profile
```

Microphone mode records through `arecord`, so a Raspberry Pi needs only
`alsa-utils` for it.

## Evaluation

Running `python evaluate.py --config <run>/args.yaml --coco-map` reports the four metrics of BatDetect2, mAP50 and mAP50-95 through pycocotools, which are the ones the paper reports.

## Development

The two contributions above live in the patched Ultralytics, not in this
repository: the positional encodings in `ultralytics/data/augment.py` and the
smoothing in `ultralytics/utils/loss.py`. To work on them without rebuilding the
image each time, create `docker-compose.override.yml`, which Compose picks up
automatically:
```yaml
services:
    yolobat:
        volumes:
            - ${ULTRALYTICS_DIR:-../yolobat-ultralytics}:/ultralytics
```
then check out Ultralytics at `7710ef05` beside this repository and apply
`ultralytics-yolobat.patch` to it.

## Citation

If you use YOLObat, please cite:

> M. Vogelbacher, H. Bellafkir, M. Fuchs, B. Freisleben. YOLObat: Real-time
> Acoustic Bat Species and Behavior Recognition via Frequency Positional
> Encodings and Taxonomy-aware Label Smoothing. Under review at *Ecological
> Informatics*, 2026.

This entry and [CITATION.cff](CITATION.cff) will be updated with the DOI once it is available.

## License

Copyright (C) 2026 Markus Vogelbacher, Hicham Bellafkir, Michael Fuchs and
Bernd Freisleben.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See [LICENSE](LICENSE) for the full text.

The pipeline runs on a patched Ultralytics, which is AGPL-3.0; both the patch
and this repository are derivative works and carry the same license.

The content of the `deploy/` directory is not a derivative work of either, and is
released under the MIT license in [deploy/LICENSE](deploy/LICENSE). Trained
models are not covered by that. Ultralytics considers models trained with its
code to be AGPL-3.0.
