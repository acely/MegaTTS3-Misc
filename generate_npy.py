import argparse
import os
import numpy as np
import torch
import librosa

# Assuming 'tts' is a top-level directory or installed package
from tts.infer_cli import MegaTTS3DiTInfer
# convert_to_wav_bytes might not be needed if librosa.load and direct processing is sufficient
# from tts.utils.audio_utils.io import convert_to_wav_bytes 
from tts.utils.commons.hparams import hparams # MegaTTS3DiTInfer likely handles hparams loading internally

def main():
    parser = argparse.ArgumentParser(description="Generate NPY latent from WAV using MegaTTS3DiTInfer's VAE.")
    parser.add_argument("--input_wav", required=True, help="Path to the input WAV file.")
    parser.add_argument("--output_npy", required=True, help="Path to save the output NPY file.")
    parser.add_argument("--ckpt_root", default="./checkpoints", help="Path to the checkpoints directory.")
    parser.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu'). Defaults to cuda if available.")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {args.device}")

    # Initialize MegaTTS3DiTInfer
    # This will also load hparams for wavvae if checkpoints are set up correctly
    try:
        infer_ins = MegaTTS3DiTInfer(device=args.device, ckpt_root=args.ckpt_root)
    except Exception as e:
        print(f"Error initializing MegaTTS3DiTInfer: {e}")
        print("Please ensure your checkpoints are correctly structured in the --ckpt_root directory.")
        print("This includes hparams.json for both the main model and the wavvae model.")
        return

    # Crucial check for VAE encoder
    if not infer_ins.has_vae_encoder:
        print("\n" + "="*50)
        print("ERROR: WavVAE encoder model is missing or not loaded correctly.")
        print(f"The script requires the WavVAE encoder (e.g., {os.path.join(args.ckpt_root, infer_ins.hp_wavvae.get('model_name_or_path', 'wavvae'), 'model_only_last.ckpt')}).")
        print("Please ensure the checkpoint for WavVAE exists and hparams.json refers to it correctly.")
        print("="*50 + "\n")
        return

    print(f"Attempting to load WavVAE hparams from directory: {infer_ins.wavvae_exp_name}")
    
    # Load WAV file
    try:
        wav, sr = librosa.core.load(args.input_wav, sr=infer_ins.sr)
        print(f"Loaded WAV: {args.input_wav}, original SR: {sr}, target SR: {infer_ins.sr}")
    except Exception as e:
        print(f"Error loading WAV file {args.input_wav}: {e}")
        return

    # Preprocessing
    # Access win_size from the loaded hparams for wavvae
    # Ensure hp_wavvae is loaded and contains 'win_size'
    if not hasattr(infer_ins, 'hp_wavvae') or 'win_size' not in infer_ins.hp_wavvae:
        print("Error: WavVAE hparams (hp_wavvae) not loaded correctly or 'win_size' is missing.")
        print("Please check your wavvae checkpoint and its hparams.json.")
        return
        
    ws = infer_ins.hp_wavvae['win_size']
    
    # Padding logic from MegaTTS3DiTInfer.preprocess
    if len(wav) % ws < ws - 1:
        wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0)
    
    # This specific padding of 12000 seems arbitrary and might need adjustment based on model needs.
    # For now, keeping it as per the prompt.
    wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0) 
    wav = wav.astype(np.float32)

    wav_tensor = torch.FloatTensor(wav)[None].to(infer_ins.device)
    print(f"Preprocessing complete. Waveform shape: {wav_tensor.shape}")

    # Encode audio to latent
    try:
        with torch.inference_mode():
            vae_latent = infer_ins.wavvae.encode_latent(wav_tensor)
        print(f"VAE latent generated. Shape: {vae_latent.shape}")
    except Exception as e:
        print(f"Error during VAE encoding: {e}")
        # This might happen if wavvae component is not fully loaded or if input shape is unexpected
        return

    # Convert to NumPy and save
    vae_latent_np = vae_latent.cpu().numpy()

    try:
        np.save(args.output_npy, vae_latent_np)
        print(f"Successfully generated {args.output_npy} from {args.input_wav}")
    except Exception as e:
        print(f"Error saving NPY file to {args.output_npy}: {e}")

if __name__ == '__main__':
    main()
