import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa # Using librosa for its robust resampling, can be torchaudio.transforms.Resample too

# Corrected import path for Audio2Mel
from tts.modules.wavvae.decoder.hifigan_modules import Audio2Mel


class FinetuneDataset(Dataset):
    def __init__(self, hparams, data_dir,_logger=None):
        self.hparams = hparams
        self.data_dir = data_dir

        self.logger = _logger if _logger else print # Basic logger

        # Check for essential hparams keys
        required_hparams = [
            'audio_sample_rate', 'segment_size', 'num_mels',
            'n_fft', 'hop_size', 'win_size', 'fmin', 'fmax'
        ]
        for key in required_hparams:
            if not hasattr(hparams, key):
                raise ValueError(f"Essential hparam '{key}' missing from configuration.")

        self.sampling_rate = hparams.audio_sample_rate
        self.segment_size = hparams.segment_size

        self.audio_files = []
        for speaker_dir in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker_dir)
            if os.path.isdir(speaker_path):
                for filename in os.listdir(speaker_path):
                    if filename.endswith(".wav"):
                        wav_path = os.path.join(speaker_path, filename)
                        lab_path = os.path.join(speaker_path, filename.replace(".wav", ".lab"))
                        if os.path.exists(lab_path): # Ensure corresponding .lab file exists
                            self.audio_files.append((wav_path, lab_path))
                        else:
                            self.logger(f"Warning: .lab file not found for {wav_path}, skipping.")

        if not self.audio_files:
            raise ValueError(f"No matching .wav/.lab pairs found in subdirectories of {data_dir}")

        self.logger(f"Found {len(self.audio_files)} audio/text pairs.")

        # Initialize Audio2Mel for mel spectrogram generation
        # Parameters for Audio2Mel should be in hparams
        self.audio_to_mel = Audio2Mel(
            n_fft=hparams.n_fft, # also known as fft_size in some contexts
            hop_size=hparams.hop_size,
            win_size=hparams.win_size,
            sampling_rate=self.sampling_rate,
            n_mels=hparams.num_mels, # also known as audio_num_mel_bins
            fmin=hparams.fmin,
            fmax=hparams.fmax
        ).float() # Ensure it's float

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path, _ = self.audio_files[idx] # lab_path is not used for now, but good to have

        try:
            # Load audio file
            audio, sr = torchaudio.load(wav_path)
        except Exception as e:
            self.logger(f"Error loading audio file {wav_path}: {e}")
            # Return a dummy item or skip. For now, let's re-raise or return None
            # To make it robust, one might try loading next item or return dummy.
            # For this example, we'll try getting the next valid item if possible, or raise error.
            if len(self) > 1: # if there are other items
                return self.__getitem__( (idx + 1) % len(self) )
            else:
                raise RuntimeError(f"Failed to load the only audio file: {wav_path}") from e


        # Resample if necessary
        if sr != self.sampling_rate:
            # Using librosa for resampling as it's often more robust for various sample rates
            # audio_np = audio.numpy()
            # audio_resampled_np = librosa.resample(audio_np[0], orig_sr=sr, target_sr=self.sampling_rate)
            # audio = torch.from_numpy(audio_resampled_np).unsqueeze(0)
            # Using torchaudio for consistency if preferred:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            audio = resampler(audio)


        # Ensure single channel (mono)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = audio.squeeze(0) # Remove channel dim for segmenting, will be (T)

        # Random segmentation
        if audio.size(0) >= self.segment_size:
            max_audio_start = audio.size(0) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio_segment = audio[audio_start : audio_start + self.segment_size]
        else:
            # Pad if audio is shorter than segment_size
            audio_segment = F.pad(audio, (0, self.segment_size - audio.size(0)), 'constant')

        # Add channel dimension back for Audio2Mel: (1, T)
        audio_segment_unsqueezed = audio_segment.unsqueeze(0)

        # Generate mel-spectrogram
        # Audio2Mel expects input shape (B, T) or (T) if B=1 internally
        # Based on hifigan_modules.Audio2Mel, it expects (N, T)
        try:
            mel_spectrogram = self.audio_to_mel(audio_segment_unsqueezed) # Mel shape (N, n_mels, T_mel)
        except Exception as e:
            self.logger(f"Error generating mel for {wav_path} (segment shape {audio_segment_unsqueezed.shape}): {e}")
            if len(self) > 1:
                 return self.__getitem__( (idx + 1) % len(self) ) # Try next
            else:
                raise RuntimeError(f"Failed to generate mel for the only audio file: {wav_path}") from e


        # Squeeze the batch dimension from mel if Audio2Mel adds it and loader doesn't expect it
        mel_spectrogram = mel_spectrogram.squeeze(0) # Shape (n_mels, T_mel)

        return mel_spectrogram, audio_segment


if __name__ == '__main__':
    import torch.nn.functional as F # For padding in main example
    import tempfile
    import soundfile as sf # To create dummy wav files
    import argparse

    print("Running FinetuneDataset example...")

    # Create dummy hparams (namespace or dict-like)
    # These should match what the dataset expects, e.g., from a config file
    # Parameters for 44.1kHz
    hparams = argparse.Namespace(
        audio_sample_rate=44100,
        segment_size=8192,         # Roughly 0.18s at 44.1kHz
        num_mels=80,               # Typical number of Mel bands
        n_fft=2048,                # FFT size, common for 44.1kHz (e.g. 2048 or 4096)
        hop_size=256,              # Hop size, determines time resolution of mel (e.g. 256 or 512)
        win_size=1024,             # Window size (e.g. 1024 or 2048)
        fmin=0.0,                  # Minimum frequency for Mel filterbank
        fmax=8000.0,               # Maximum frequency for Mel filterbank (Nyquist is 22050 for 44.1kHz)
                                   # Often set lower, e.g. 8000 for speech
        # These might also be needed by other parts of a training script
        batch_size=4,
        learning_rate=1e-4,
    )

    # Adjust fmax to be more realistic for 44.1kHz if full bandwidth is desired
    hparams.fmax = hparams.audio_sample_rate / 2 # Nyquist

    # Create a temporary directory for dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy speaker directories and files
        speaker1_dir = os.path.join(tmpdir, "Speaker-1")
        speaker2_dir = os.path.join(tmpdir, "Speaker-2")
        os.makedirs(speaker1_dir, exist_ok=True)
        os.makedirs(speaker2_dir, exist_ok=True)

        # Create dummy .wav and .lab files
        # File 1 (longer than segment_size)
        sf.write(os.path.join(speaker1_dir, "file1.wav"), torch.randn(hparams.audio_sample_rate * 2).numpy(), hparams.audio_sample_rate)
        with open(os.path.join(speaker1_dir, "file1.lab"), "w") as f:
            f.write("This is a dummy transcription for file1.")

        # File 2 (shorter than segment_size, will be padded)
        sf.write(os.path.join(speaker1_dir, "file2.wav"), torch.randn(hparams.segment_size // 2).numpy(), hparams.audio_sample_rate)
        with open(os.path.join(speaker1_dir, "file2.lab"), "w") as f:
            f.write("This is a dummy transcription for file2.")

        # File 3 (different speaker)
        sf.write(os.path.join(speaker2_dir, "file3.wav"), torch.randn(hparams.audio_sample_rate * 1).numpy(), hparams.audio_sample_rate)
        with open(os.path.join(speaker2_dir, "file3.lab"), "w") as f:
            f.write("This is a dummy transcription for file3.")

        # File 4 (will have missing .lab to test skipping)
        sf.write(os.path.join(speaker2_dir, "file4_no_lab.wav"), torch.randn(hparams.audio_sample_rate * 1).numpy(), hparams.audio_sample_rate)


        print(f"Dummy data created in {tmpdir}")

        try:
            dataset = FinetuneDataset(hparams, tmpdir)
            print(f"Dataset initialized. Number of items: {len(dataset)}")

            if len(dataset) > 0:
                # Get a sample
                mel, audio = dataset[0]
                print(f"Sample 0 - Mel shape: {mel.shape}, Audio shape: {audio.shape}")
                assert mel.shape[0] == hparams.num_mels
                assert audio.shape[0] == hparams.segment_size

                # Expected mel time frames: segment_size / hop_size
                # T_mel = segment_size / hop_size. Audio2Mel pads for n_fft so it might be slightly different.
                # For HiFiGAN's Audio2Mel, it's typically (segment_size // hop_size) + some padding related to n_fft and centering
                # Let's check it's close to segment_size / hop_size
                expected_mel_len = hparams.segment_size // hparams.hop_size
                print(f"Expected mel length around: {expected_mel_len}, Actual: {mel.shape[1]}")
                # Audio2Mel from HiFiGAN pads the input so output length is ceil(input_length / hop_size)
                # If input_length is segment_size, then T_mel = ceil(segment_size / hop_size)
                # It also pads input with n_fft // 2 on each side if pad_center=True (default)
                # which means the effective length for mel calculation is segment_size + n_fft
                # So, T_mel should be (segment_size + n_fft - win_size) // hop_size + 1 if win_size is used for framing
                # Or more simply, for HiFiGAN's Audio2Mel, it seems to be (segment_size // hop_size)
                # For pad_short=True and centered=True (default), it pads to be multiple of hop_size
                # and pads by n_fft//2 at ends.
                # The output length of mel from Audio2Mel is `(input_length // self.hop_size)` if not centered
                # or `(input_length + self.n_fft - self.hop_size) // self.hop_size` if centered.
                # Given Audio2Mel defaults, it pads input by `n_fft // 2` on both sides.
                # So effective input is `segment_size + n_fft`. Mel frames: `(segment_size + n_fft) // hop_size`.
                # This needs to be precise if used in model. For now, a rough check is fine.
                # The hifigan_modules.Audio2Mel does:
                # audio = torch.nn.functional.pad(audio, (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)), mode='reflect')
                # This padding makes the number of frames = audio_length / hop_size
                # So, for input `self.segment_size`, output mel frames should be `self.segment_size / self.hop_size`
                assert mel.shape[1] == expected_mel_len

                print("Dataset sample check passed.")

                # Example of iterating through DataLoader
                from torch.utils.data import DataLoader
                dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True)
                for i, batch in enumerate(dataloader):
                    mels, audios = batch
                    print(f"Batch {i} - Mels shape: {mels.shape}, Audios shape: {audios.shape}")
                    if i >= 2: # Print a few batches
                        break
                print("DataLoader iteration example passed.")

            else:
                print("Dataset is empty, cannot test sample retrieval.")

        except Exception as e:
            print(f"An error occurred during dataset test: {e}")
            import traceback
            traceback.print_exc()

    print("FinetuneDataset example finished.")
