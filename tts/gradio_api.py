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

import multiprocessing as mp
import torch
import os
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def scan_assets_directory():
    """
    Scans the 'assets' directory (expected at the project root) for .wav and .npy files.

    Returns:
        tuple: (wav_files, npy_files)
            wav_files (list): List of relative paths to .wav files (e.g., 'assets/audio1.wav').
            npy_files (list): List of relative paths to .npy files (e.g., 'assets/prompt1.npy').
    """
    # Assuming gradio_api.py is in 'tts/' subdirectory, so project root is one level up.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir_path = os.path.join(project_root, 'assets')

    wav_files = []
    npy_files = []

    if not os.path.exists(assets_dir_path) or not os.path.isdir(assets_dir_path):
        print(f"Assets directory not found at: {assets_dir_path}")
        return wav_files, npy_files

    for filename in os.listdir(assets_dir_path):
        # Construct the relative path from the project root for UI display and potential use.
        # The 'assets' part is added to make it e.g. "assets/filename.ext"
        relative_path = os.path.join('assets', filename)
        if filename.lower().endswith(".wav"):
            wav_files.append(relative_path)
        elif filename.lower().endswith(".npy"):
            npy_files.append(relative_path)
    
    wav_files.sort()
    npy_files.sort()
    
    return wav_files, npy_files


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_npy_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=inp_npy_path)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)


def main(prompt_wav_selection, custom_wav_upload, prompt_npy_selection, custom_npy_upload, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    actual_audio_path = None
    if prompt_wav_selection == "Upload custom WAV...":
        actual_audio_path = custom_wav_upload # This will be the filepath from gr.Audio
    elif prompt_wav_selection is not None: # A preset WAV file is selected
        actual_audio_path = prompt_wav_selection
    
    if actual_audio_path is None:
        # This check is important because the model_worker expects an audio path.
        raise gr.Error("Prompt WAV is required. Please select from the dropdown or upload a custom file.")

    actual_npy_path = None
    if prompt_npy_selection == "Upload custom NPY...":
        actual_npy_path = custom_npy_upload # This will be the filepath from gr.File
    elif prompt_npy_selection == "None (use VAE encoder if available)":
        actual_npy_path = None # Explicitly set to None
    elif prompt_npy_selection is not None: # A preset NPY file is selected
        actual_npy_path = prompt_npy_selection
    # If prompt_npy_selection was None (no dropdown selection) and custom_npy_upload is also None, actual_npy_path remains None.

    print(f"Push task to the inp queue | Audio: {actual_audio_path}, NPY: {actual_npy_path}, Text: {inp_text}, Timestep: {infer_timestep}, PW: {p_w}, TW: {t_w}")
    input_queue.put((actual_audio_path, actual_npy_path, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        print("")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    wav_files, npy_files = scan_assets_directory()
    # Add default options for uploading, ensuring they are distinct
    # and will be handled in the main function.
    wav_choices = ["Upload custom WAV..."] + wav_files
    npy_choices = ["Upload custom NPY...", "None (use VAE encoder if available)"] + npy_files

    print("Start open workers")
    for i in range(num_workers):
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None))
        p.start()
        processes.append(p)

    api_interface = gr.Interface(fn=
                                partial(main, processes=processes, input_queue=input_queue, 
                                        output_queue=output_queue), 
                                inputs=[
                                    gr.Dropdown(choices=wav_choices, value="Upload custom WAV...", label="Prompt WAV"),
                                    gr.Audio(type="filepath", label="Custom WAV Upload (if selected above)"),
                                    gr.Dropdown(choices=npy_choices, value="Upload custom NPY...", label="Prompt NPY (optional)"),
                                    gr.File(type="filepath", label="Custom NPY Upload (if selected above)"),
                                    "text", 
                                    gr.Number(label="infer timestep", value=32),
                                    gr.Number(label="Intelligibility Weight", value=1.4),
                                    gr.Number(label="Similarity Weight", value=3.0)
                                ], 
                                outputs=[gr.Audio(label="Synthesized Audio")],
                                title="MegaTTS3",  
                                description="Upload a speech clip as a reference for timbre, " +
                                "upload the pre-extracted latent file, "+
                                "input the target text, and receive the cloned voice.", concurrency_limit=1)
    api_interface.launch(server_name='0.0.0.0', server_port=7929, debug=True)
    for p in processes:
        p.join()
