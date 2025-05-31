I've identified the source of the 24kHz audio output limitation in your TTS project. 

My investigation has concluded. 

The 24kHz sampling rate is primarily due to explicit settings in the configuration files. 

Key findings: 
- `checkpoints/diffusion_transformer/config.yaml` specifies `audio_sample_rate: 24000`. 
- `checkpoints/wavvae/config.yaml` specifies `audio_sample_rate: 24000` and its nested `melgan_config` also specifies `sampling_rate: 24000`. 
- Your codebase, including files like `tts/utils/audio_utils/io.py`, uses this 24kHz rate, often as a default or hardcoded value.

This confirms that the limitation is a design choice originating from these configurations. 

Modifying this rate would likely require retraining or fine-tuning the existing models for optimal performance at a different sample rate.

The WavVAE decoder (based on HiFi-GAN) takes the generated acoustic latents (concatenated with the prompt latent for seamless transition) and synthesizes the final output waveform.
