# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F

# Imports for type hinting or if used outside conditional blocks can remain
# from tts.modules.wavvae.decoder.seanet_encoder import Encoder 
# from tts.modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
# from tts.modules.wavvae.decoder.hifigan_modules import Generator, Upsample


class WavVAE_V3(nn.Module):
    def __init__(self, hparams=None, init_encoder=True, init_decoder=True):
        super().__init__()
        self.init_encoder = init_encoder
        self.init_decoder = init_decoder
        self.hparams = hparams

        if self.init_encoder:
            from tts.modules.wavvae.decoder.seanet_encoder import Encoder
            self.encoder = Encoder(dowmsamples=[6, 5, 4, 4, 2])
            self.proj_to_z = nn.Linear(512, 64)
        else:
            self.encoder = None
            self.proj_to_z = None

        if self.init_decoder:
            self.proj_to_decoder = nn.Linear(32, 320)
            
            if self.hparams is None:
                raise ValueError("hparams must be provided if init_decoder is True")
            config_path = self.hparams['melgan_config']
            
            import argparse # Local import
            args = argparse.Namespace()
            args.__dict__.update(config_path)

            from tts.modules.wavvae.decoder.hifigan_modules import Generator, Upsample
            self.latent_upsampler = Upsample(320, 4)
            self.decoder = Generator(
                input_size_=args.n_mel_channels,
                ngf=args.ngf if hasattr(args, 'ngf') else 128,
                n_residual_layers=args.n_residual_layers,
                num_band=args.num_band if hasattr(args, 'num_band') else 1,
                args=args,
                ratios=args.up_sample
            )
        else:
            self.proj_to_decoder = None
            self.latent_upsampler = None
            self.decoder = None

    ''' encode waveform into 25 hz latent representation '''
    def encode_latent(self, audio):
        if not self.init_encoder or self.encoder is None:
            raise RuntimeError("Encoder was not initialized for WavVAE_V3. Cannot call encode_latent.")
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio):
        if not self.init_encoder or self.encoder is None or self.proj_to_z is None:
            raise RuntimeError("Encoder was not initialized. Cannot call encode.")
        from tts.modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution # Import locally
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        posterior = DiagonalGaussianDistribution(x) # Corrected variable name
        return posterior

    def decode(self, latent):
        if not self.init_decoder or self.decoder is None or self.proj_to_decoder is None or self.latent_upsampler is None:
            raise RuntimeError("Decoder was not initialized. Cannot call decode.")
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio):
        if not self.init_encoder or not self.init_decoder:
            raise RuntimeError("Both encoder and decoder must be initialized for a full forward pass.")
        if self.encoder is None or self.decoder is None: # Additional check for components
             raise RuntimeError("Encoder or Decoder components are None. Both must be initialized for a full forward pass.")

        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)
        return recon_wav, posterior