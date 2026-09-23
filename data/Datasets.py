import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace


from ultralytics.utils.instance import Instances
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import math
import librosa
from copy import deepcopy
import audiomentations as AM
from util.add_background_noise import CustomAddBackgroundNoise

from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    v8_transforms,
)


def compute_pos_enc_channels(H: int, W: int, lin: bool = False) -> list[np.ndarray]:
    """Sinusoidal frequency positional encoding channels for an (H, W) image."""
    freq = np.linspace(0, 1, H, endpoint=False, dtype=np.float32).reshape(H, 1)
    if lin:
        # Linear frequency axis, same value in both channels
        enc = np.repeat(freq, W, axis=1) * 255  # [H, W], range [0, 255]
        return [enc, enc]
    # Sinusoidal encoding: sin and cos of normalised frequency axis
    sin_ch = np.repeat(((np.sin(2 * np.pi * freq) + 1.0) * 127.5), W, axis=1)
    cos_ch = np.repeat(((np.cos(2 * np.pi * freq) + 1.0) * 127.5), W, axis=1)
    return [sin_ch, cos_ch]


def create_spectrogram(
    wav: np.ndarray, hyp: SimpleNamespace, low_cutoff_bins: int
) -> np.ndarray:
    if len(wav) != (hyp.sr * hyp.analyze_length):
        wav = librosa.resample(
            wav, orig_sr=len(wav), target_sr=(hyp.sr * hyp.analyze_length)
        )[:int(hyp.sr * hyp.analyze_length)] # resampling sometimes off by one, e.g. returning 384001 instead of 384000
        
    # Compute STFT using librosa
    stft_result: np.ndarray = librosa.stft(
        wav,
        n_fft=hyp.n_fft,
        hop_length=hyp.hop_length,
        window=hyp.window_func,
        center=True,
        pad_mode="reflect",
    )
    
    spec: np.ndarray = np.abs(stft_result)
    spec: np.ndarray = spec[::-1, :]  # high freq → row 0

    spec_db: np.ndarray = librosa.amplitude_to_db(spec, ref=np.max)
    spec_db = min_max_normalization(spec_db) * 255
    spec_db = spec_db[: spec_db.shape[0] - low_cutoff_bins, :]

    H, W = spec_db.shape

    channels = [spec_db]

    # Positional encoding channels (sin/cos or linear frequency axis)
    if hyp.pos_enc:
        channels += compute_pos_enc_channels(H, W, getattr(hyp, 'lin', False))

    # Without the positional encodings the spectrogram is replicated to three
    # channels, so the input is always 3xHxW.
    if len(channels) == 1:
        channels = [spec_db, spec_db, spec_db]

    return np.stack(channels, axis=-1).astype(np.uint8)


def load_audio_segment(
    path: str, sr: int, time_index: float, duration: float = 1.0
) -> np.ndarray:
    """
    Loads Audio segment directly from the file.

    :param path: path of audio file
    :param sr: wanted sample rate
    :param time_index: start time of audio segment
    :param duration: duration of segment
    :return: audio segment with length of duration
    """
    wav, _ = librosa.load(path, sr=sr, offset=time_index, duration=duration)


    wav = np.pad(
        wav,
        (0, max(0, int(sr * duration) - wav.shape[-1])),
        mode="constant",
        constant_values=[0],
    )

    return wav


def clip_bbox(
    start_time: float, end_time: float, box: np.ndarray, sr: float, low_cutoff_hz: int
):
    x, y, width, height = box
    clipped = False
    # check if start or end of the bbox is in the frame
    if start_time <= x < end_time or start_time <= x + width < end_time:
        # if bbox starts before the frame,
        # set x to start of the frame and reduce width
        if x < start_time:
            width = width - (start_time - x)
            x = start_time
            clipped = True
        # if bbox ends after the frame,
        # set width to end of the frame
        if x + width > end_time:
            width = end_time - x
            clipped = True
        if width < 0.003 and clipped:
            box = None  # type: ignore
    else:
        box = None  # type: ignore

    # for cutting off frequencies below cut_low_freq
    if y < low_cutoff_hz:
        diff = low_cutoff_hz - y
        y = y + diff
        height = height - diff
    # if bbox y+height is bigger than the max frequency,
    # reduce height so the bbox reaches the max frequency
    height: float = min(height, sr / 2 - y)
    y = y - low_cutoff_hz
    bbox = (x, y, width, height) if box else None
    return clipped, bbox


def adapt_bbox(
    box: np.ndarray,
    class_label: str,
    frame: tuple[float, float],
    sr: int,
    analyze_length: float,
    clip_bboxes: bool = True,
    low_cutoff_hz: int = 0,
):
    """
    Adapts the bbox to the yolo format and checks if it is in the time frame

    :param box: bbox
    :param frame: timeframe (x,y)
    :param sr: samplerate
    :return: adapted bbox [x,y,width,height]
    """
    start_time, end_time = frame
    clipped: bool = False
    if clip_bboxes:
        clipped, box = clip_bbox(start_time, end_time, box, sr, low_cutoff_hz)  # type: ignore
    if not box:
        return None

    x, y, width, height = box

    if height < 4000:
        diff = (4000 - height) / 2
        y = y - diff
        height = 4000


    x_relative = x - start_time

    x_center = x_relative + (width / 2)
    x_center_norm = x_center / analyze_length
    width_norm: float = width / analyze_length

    y_center = y + (height / 2)
    freq_max: float = (sr / 2) - low_cutoff_hz
    y_center_norm = y_center / freq_max
    height_norm: float = height / freq_max

    y_center_norm = (
        1.0 - y_center_norm
    )  # swap because 0 is top of image and bottom is 1

    if 0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1:
        return [x_center_norm, y_center_norm, width_norm, height_norm]
    else:
        return None


def read_file_as_array(file_path):
    """
    Read strings from .txt file

    :param file_path: file path
    :return: array of strings from the file
    """
    with open(file_path, "r") as file:
        lines = file.read().splitlines()

    return lines


def min_max_normalization(spec):
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max == spec_min:
        return np.zeros_like(spec)
    return (spec - spec_min) / (spec_max - spec_min)


def calculate_ratio_pad(original_size, target_size):
    original_width, original_height = original_size
    target_width, target_height = target_size

    ratio = min(target_width / original_width, target_height / original_height)

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    pad_width = (target_width - new_width) / 2
    pad_height = (target_height - new_height) / 2

    ratio_pad = ((ratio, ratio), (pad_width, pad_height))

    return ratio_pad


def calculate_weights(cls, label_weights, agg_func):
    """
    Calculate the aggregated weight for each label based on class weights.

    :return:
        float: A weight value corresponding to the item.
    """
    if label_weights is None:
        return 1

    cls = np.asarray(cls, dtype=int)
    cls = cls.reshape(-1).astype(int)

    if len(cls) == 0:
        return 1

    return agg_func(label_weights[cls])


def calculate_probabilities(weights):
    """
    Calculate and store the sampling probabilities based on the weights.

    Returns:
        list: A list of sampling probabilities corresponding to each label.
    """
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return probabilities


NOISE_TYPES = ("crickets",)
NOISE_DIR = Path(__file__).resolve().parent.parent / "files" / "noise"


def load_noise_files() -> dict[str, list[str]]:
    """Read files/noise/<type>.txt: one recording per line, relative to audio_path.

    Blank lines and lines starting with # are ignored. These recordings are mixed
    in by the background-noise augmentation and are therefore held out of the
    training set. Entries missing from disk are dropped by audiomentations when
    the transform is built, so an incomplete copy of the data just yields a
    smaller pool.
    """
    noise_files = {}
    for noise_type in NOISE_TYPES:
        path = NOISE_DIR / f"{noise_type}.txt"
        lines = path.read_text().splitlines() if path.exists() else []
        noise_files[noise_type] = [
            line.strip() for line in lines
            if line.strip() and not line.startswith("#")
        ]
    return noise_files


class YoloDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        filter_file=None,
        augment=True,
        hyp: SimpleNamespace = None,  # type: ignore
        audio_path="/data/bats/",
    ):  # type: ignore
        """
        :param annotation_file: file with annotations, you can find the script to create these in the repo.
        :param filter_file: file, which holds the file paths of files which should be used in the dataset
        :param audio_path: file path prefix for audio files
        """

        self.n_fft = hyp.n_fft
        self.hop_length = hyp.hop_length
        self.sr = hyp.sr
        self.analyze_length = hyp.analyze_length
        self.shape = []
        x_size = (
            math.floor(
                math.ceil((self.sr / self.hop_length) * self.analyze_length) / 32
            )
            * 32
        )
        self.low_cutoff_bins = 0
        if hyp.cut_low_freq:
            self.low_cutoff_bins = math.floor(
                hyp.cut_low_freq / ((self.sr / 2) / (self.n_fft / 2 + 1))
            )
            hyp.cut_low_freq = self.low_cutoff_bins * int(
                (self.sr / 2) / (self.n_fft / 2 + 1)
            )
        y_size = math.floor(((self.n_fft / 2 + 1) - self.low_cutoff_bins) / 32) * 32
        self.shape = [y_size, x_size]
        self.agg_func = np.max

        self.audio_path = audio_path
        self.noise_files = load_noise_files() if augment else {}
        # the background-noise recordings are augmentation material, so they are
        # filtered out of the training set below
        self.noise_file_set = {f for files in self.noise_files.values() for f in files}

        with open(annotation_file, "r") as file:
            data = json.load(file)

        self.classes = {}
        self.class_map = {}
        self.data = {}

        if hyp.single_cls:
            self.classes = {0: "bat"}
            self.class_map = {0: 0}
            self.data["names"] = self.classes
            self.data["nc"] = 1
        else:
            i = 0
            for x in data["categories"]:
                self.classes[i] = x["name"]
                self.class_map[x["id"]] = i
                i += 1
            self.data = {
                "nc": len(self.classes),
                "names": self.classes,
            }

        images = {}
        print(len(data["images"]), "images found in dataset")
        for image in tqdm(data["images"]):
            if augment and image["file_name"] in self.noise_file_set:
                # held out for the AddBackgroundNoise augmentation
                continue
            # header duration can be wrong for some files
            duration = librosa.get_duration(path=audio_path + image["file_name"])
            if duration > 1000:  # implausible, the file header is broken
                print("Error in file length", image["file_name"])
                duration = 5
            images[image["id"]] = {
                "path": audio_path + image["file_name"],
                "duration": duration,
                "cls": [],
                "bboxes": [],
            }

        for annotation in data["annotations"]:
            if annotation["image_id"] in images:
                image_id = annotation["image_id"]
                category_id = 0 if hyp.single_cls else annotation["category_id"]

                mapped_class_id = self.class_map[category_id]

                images[image_id]["cls"].append([mapped_class_id])

                images[image_id]["bboxes"].append(annotation["bbox"])

        if filter_file is not None:
            # One recording per line, relative to audio_path.
            allowed = {audio_path + path for path in read_file_as_array(filter_file)}
            images = {
                key: label
                for key, label in images.items()
                if label["path"] in allowed
            }

        self.images = images
        print(len(self.images), "images after filtering")
        self.augment = augment
        self.hyp: SimpleNamespace = hyp

        self.label_weights = self.calculate_label_weights(hyp)
        self.labels, self.label_weights = (
            self.get_snippets(images)
            if hyp.snippets and self.augment
            else self.get_labels(augment, images)
        )
        if self.hyp.oversampling and self.augment:
            self.probabilities = calculate_probabilities(self.label_weights)

        self.imgsz = self.shape
        self.hyp.imgsz = self.shape
        self.ni = (
            len(self.labels.keys())  # type: ignore
            if self.hyp.snippets and self.augment
            else len(self.labels)
        )  # type: ignore

        self.task = hyp.task
        self.rect = hyp.rect
        self.use_segments = self.task == "segment"
        self.use_keypoints = self.task == "pose"
        self.use_obb = self.task == "obb"
        self.batch_size = self.hyp.batch

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = (
            min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        )

        self.transforms = self.build_transforms(self.hyp)

        self.waveform_transforms = (
            self.build_waveform_transforms(self.hyp) if augment else None
        )

        super(YoloDataset, self).__init__()

    def window_starts(self, label: dict[str, Any]) -> list[float]:
        """Start times of the windows the recording is tiled into."""
        n = math.ceil(label["duration"] / self.analyze_length)
        return [i * self.analyze_length for i in range(n)]

    def close_mosaic(self, hyp: dict) -> None:
        """Turn off mosaic and mixup for the final epochs.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.mixup = 0.0
        self.transforms = self.build_transforms(hyp)

    def calculate_label_weights(self, hyp):
        """Per-class sampling weight, inversely proportional to how often the class occurs.

        Counted over the annotations this dataset holds, so no external table is
        needed. Only the ratios matter: the weights are normalised by
        calculate_probabilities before they are used, so any common factor cancels.
        That is also why "balanced" and "unbalanced" are the same setting -- they
        differ by a factor of len(classes).
        """
        if hyp.label_weights not in ("balanced", "unbalanced"):
            return None
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for label in self.images.values():
            cls = np.asarray(label["cls"], dtype=int).reshape(-1)
            np.add.at(counts, cls, 1)
        if (counts == 0).any():
            return None  # a class with no annotations here -- fall back to equal weights
        return counts.sum() / counts

    def get_snippets(self, labels: dict[str, Any]) -> tuple[dict[int, Any], list[Any]]:
        segment_dict = {}
        index = 0
        weights = []
        for id in labels.keys():
            segments_per_audio = max(
                1,
                math.ceil(len(self.window_starts(self.images[id])) * self.hyp.snippet_ratio),
            )
            for _ in range(math.ceil(segments_per_audio)):
                if self.hyp.oversampling and self.augment:
                    weights.append(
                        calculate_weights(
                            self.images[id]["cls"], self.label_weights, self.agg_func
                        )
                        / segments_per_audio
                    )
                segment_dict[index] = id
                index = index + 1
        return segment_dict, weights

    def get_labels(
        self, augment: bool, labels: dict[str, Any]
    ) -> tuple[list[Any], list[Any]]:
        label_items = []
        weights = []

        for _, item_label in tqdm(labels.items()):
            for start_time in self.window_starts(item_label):
                ori_shape = self.shape
                resized_shape = self.shape

                ratio_pad = calculate_ratio_pad(ori_shape, resized_shape)

                bboxes = []
                cls = []
                for i in range(len(item_label["bboxes"])):
                    box = item_label["bboxes"][i]
                    class_label = item_label["cls"][i]
                    adapted_box = adapt_bbox(
                        box,
                        self.classes[class_label[0]],
                        (start_time, start_time + self.analyze_length),
                        self.sr,
                        self.analyze_length,
                        self.hyp.clip_bboxes,
                        self.hyp.cut_low_freq,
                    )
                    if adapted_box is not None:
                        bboxes.append(adapted_box)
                        cls.append(class_label)

                if len(cls) == 0:
                    if augment:
                        continue  # all seconds without bbox in training will be omitted
                    cls = torch.zeros((0, 1), dtype=torch.int32).numpy()
                    bboxes = torch.zeros((0, 4)).numpy()
                else:
                    cls = torch.tensor(
                        cls, dtype=torch.int32
                    ).numpy()
                    bboxes = torch.tensor(bboxes, dtype=torch.float32).numpy()
                label_items.append(
                    {
                        "cls": cls,
                        "bboxes": bboxes,
                        "im_file": item_label["path"],
                        "ori_shape": tuple(ori_shape),
                        "resized_shape": tuple(resized_shape),
                        "ratio_pad": ratio_pad,
                        "time": start_time,
                        "segments": [],  # add for augmentations
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
                if self.hyp.oversampling and self.augment:
                    weights.append(
                        calculate_weights(cls, self.label_weights, self.agg_func)
                    )

        return label_items, weights

    def __getitem__(self, index):
        """
        Generates spec and returns it with the corresponding label
        :param index: index
        :return: item_label with spectrogram
        """
        if self.augment and self.hyp.oversampling:
            index = np.random.choice(len(self.labels), p=self.probabilities)
        if self.hyp.snippets and self.augment:
            item_label = self._get_image_and_label(index)
        else:
            item_label = deepcopy(self.labels[index])
        self.buffer.append(index)
        if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
            self.buffer.pop(0)

        wav = load_audio_segment(
            item_label["im_file"], self.sr, item_label["time"], self.analyze_length
        )

        item_label["img"] = wav
        if self.augment and self.hyp.augment_audio:
            item_label["img"] = self.waveform_transforms(  # type: ignore
                item_label["img"], sample_rate=self.sr
            )
        item_label["img"] = create_spectrogram(
            item_label["img"], self.hyp, self.low_cutoff_bins
        )
        item_label = self.transforms(self.update_labels_info(item_label))
        
        item_label["img"] = item_label["img"].float()

        return item_label

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        if self.hyp.snippets and self.augment:
            label = self._get_image_and_label(index)
        else:
            label = deepcopy(
                self.labels[index]
            )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        wav = load_audio_segment(
            label["im_file"], self.sr, label["time"], self.analyze_length
        )

        label["img"] = create_spectrogram(wav, self.hyp, self.low_cutoff_bins)
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]  # type: ignore
        return self.update_labels_info(label)

    def _get_image_and_label(self, index):
        image_id = self.labels[index]
        item_label = deepcopy(self.images[image_id])
        # a snippet starts anywhere that leaves analyze_length seconds of audio
        last_start = max(0.0, item_label["duration"] - self.analyze_length)
        item_label["time"] = np.random.uniform(0.0, last_start) if last_start else 0.0
        ori_shape = self.shape
        resized_shape = self.shape

        ratio_pad = calculate_ratio_pad(ori_shape, resized_shape)

        bboxes = []
        cls = []
        for i in range(len(item_label["bboxes"])):
            box = item_label["bboxes"][i]
            class_label = item_label["cls"][i]
            adapted_box = adapt_bbox(
                box,
                self.classes[class_label[0]],
                (item_label["time"], item_label["time"] + self.analyze_length),
                self.sr,
                self.analyze_length,
                self.hyp.clip_bboxes,
                self.hyp.cut_low_freq,
            )
            if adapted_box is not None:
                bboxes.append(adapted_box)
                cls.append(class_label)

        if len(cls) == 0:
            cls = torch.zeros((0, 1), dtype=torch.int32).numpy()
            bboxes = torch.zeros((0, 4)).numpy()
        else:
            cls = torch.tensor(cls, dtype=torch.int32).numpy()
            bboxes = torch.tensor(bboxes, dtype=torch.float32).numpy()
        return {
            "cls": cls,
            "bboxes": bboxes,
            "im_file": item_label["path"],
            "ori_shape": tuple(ori_shape),
            "resized_shape": tuple(resized_shape),
            "ratio_pad": ratio_pad,
            "time": item_label["time"],
            "segments": [],  # add for augmentations
            "normalized": True,
            "bbox_format": "xywh",
        }

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format", "xywh")
        normalized = label.pop("normalized", True)

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(
            bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized
        )
        return label

    def __len__(self):
        return (
            len(self.labels.keys())  # type: ignore
            if self.hyp.snippets and self.augment
            else len(self.labels)
        )  # type: ignore

    def build_waveform_transforms(self, hyp: SimpleNamespace):
        """Waveform augmentations: background noise, then a random gain.

        The noise pool is skipped when files/noise/crickets.txt lists no
        recordings, so training works without one.
        """
        transforms = []
        crickets = [self.audio_path + f for f in self.noise_files["crickets"]]
        if crickets:
            transforms.append(
                CustomAddBackgroundNoise(crickets, p=hyp.crickets_noise, hyp=hyp)
            )
        transforms.append(
            AM.Gain(min_gain_db=hyp.min_gain_db, max_gain_db=hyp.max_gain_db, p=hyp.gain)
        )
        return AM.Compose(transforms=transforms)

    def build_transforms(self, hyp: SimpleNamespace):
        """Builds and appends transforms to the list."""
        if self.augment and hyp.augment_spectrogram:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([])
        transforms.append(
            LetterBox(new_shape=self.shape, scale_fill=True, scaleup=False)
        )
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
            )
        )
        return transforms
