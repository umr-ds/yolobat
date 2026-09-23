import random
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
import librosa
from types import SimpleNamespace

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
    find_audio_files_in_paths,
)


class CustomAddBackgroundNoise(BaseWaveformTransform):
    """Mix a random second of a background recording into the input, at a random SNR.

    Adapted from audiomentations.AddBackgroundNoise (MIT), which reads whole files;
    this one takes a random segment of the configured length instead, so the long
    recordings in the noise pool can be used without loading them completely.
    """

    def __init__(
        self,
        sounds_path: Union[List[Path], List[str], Path, str],
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
        p: float = 0.5,
        hyp: SimpleNamespace=None, # type: ignore
    ):
        """
        :param sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be background noises.
        :param min_snr_db: Minimum signal-to-noise ratio in dB
        :param max_snr_db: Maximum signal-to-noise ratio in dB
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.sounds_path = sounds_path
        self.sound_file_paths = [str(p) for p in find_audio_files_in_paths(self.sounds_path)]

        assert len(self.sound_file_paths) > 0

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        self.hyp = hyp

    def _load_sound(self, file_path, sample_rate):
        duration = librosa.get_duration(path=file_path)
        duration = duration if duration < 1000 else 5 # there are files that have an invalid duration
        time_index = np.random.uniform(0, max(0, duration - self.hyp.analyze_length))
        wav, _ = librosa.load(file_path, sr=sample_rate, offset=time_index, duration=self.hyp.analyze_length)
        wav = np.pad(wav, (0, int(sample_rate * self.hyp.analyze_length) - wav.shape[-1]), mode='constant', constant_values=[0])

        return wav

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_db"] = random.uniform(self.min_snr_db, self.max_snr_db) # type: ignore
            self.parameters["noise_file_path"] = random.choice(self.sound_file_paths) # type: ignore

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        noise_sound = self._load_sound(
            self.parameters["noise_file_path"], sample_rate
        )

        noise_rms = calculate_rms(noise_sound)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(self.parameters["noise_file_path"])
            )
            return samples

        clean_rms = calculate_rms(samples)

        desired_noise_rms = calculate_desired_noise_rms(
            clean_rms, self.parameters["snr_db"]
        )
        noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        # Repeat the sound if it shorter than the input sound
        num_samples = len(samples)
        while len(noise_sound) < num_samples:
            noise_sound = np.concatenate((noise_sound, noise_sound))

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        # Return a mix of the input sound and the background noise sound
        return samples + noise_sound
