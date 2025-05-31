import os
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# Hard-coded audio hyperparameters from whisper.audio
N_FFT = 400
HOP_LENGTH = 160

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    # This file (mel_filters.npz) needs to be manually placed here:
    # tts/utils/audio_utils/assets/mel_filters.npz
    # The path should be relative to this file (whisper_mel_utils.py)
    current_dir = os.path.dirname(__file__)
    filters_path = os.path.join(current_dir, "assets", "mel_filters.npz")

    if not os.path.exists(filters_path):
        # Attempt to create the assets directory if it doesn't exist
        assets_dir = os.path.join(current_dir, "assets")
        if not os.path.exists(assets_dir):
            try:
                os.makedirs(assets_dir)
                # This message is for the case where the subtask creates the dir
                print(f"Created directory: {assets_dir}")
                print(f"Please ensure 'mel_filters.npz' is placed in {assets_dir}")
            except OSError as e:
                raise FileNotFoundError(
                    f"'assets' directory could not be created at {assets_dir} (Error: {e}). "
                    f"Please create it manually and place 'mel_filters.npz' (from OpenAI Whisper whisper/assets/) inside it."
                ) from e

        raise FileNotFoundError(
            f"'mel_filters.npz' not found at {filters_path}. "
            f"Please download it from the OpenAI Whisper repository (whisper/assets/mel_filters.npz) "
            f"and place it in the '{os.path.join('tts', 'utils', 'audio_utils', 'assets')}' directory relative to your project root."
        )

    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f'mel_{n_mels}']).to(device)

def log_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)

    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
