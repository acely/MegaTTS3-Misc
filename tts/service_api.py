import multiprocessing as mp
import torch
import os
import traceback
from functools import partial # May not be needed initially
import uvicorn # For running the app
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse # For returning audio
import io
from typing import Optional # Added Optional

# Assuming these can be imported directly if PYTHONPATH is set,
# or adjust relative paths if necessary.
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav

# Global variables for worker management (similar to gradio_api.py)
# These will be initialized when the module is loaded.
mp_manager = None
input_queue = None
output_queue = None
processes = []
# Global variable for the inference pipe, to be initialized in a worker or on startup
# For simplicity in a REST API, the model_worker pattern from Gradio is good.

# Model worker function (adapted from gradio_api.py)
# This function will run in a separate process.
def model_worker_main(input_q, output_q, device_id, wavvae_load_mode_for_worker):
    print(f"Model worker started on device_id: {device_id} with WavVAE load mode: {wavvae_load_mode_for_worker}")
    device = None
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    
    print(f"Worker using device: {device}")

    try:
        # Pass wavvae_load_mode here
        infer_pipe = MegaTTS3DiTInfer(device=device, wavvae_load_mode=wavvae_load_mode_for_worker)
    except Exception as e:
        print(f"!!! Error initializing MegaTTS3DiTInfer in worker: {e}")
        traceback.print_exc()
        # Signal failure to parent or handle as appropriate
        # For now, worker will be alive but unable to process.
        # A more robust solution might involve a status check or specific error on queue.
        infer_pipe = None # Ensure it's None if init fails

    while True:
        task = input_q.get()
        if task is None: # Sentinel for stopping the worker
            print(f"Model worker {device_id} received stop signal.")
            break
        
        if infer_pipe is None:
            output_q.put({"error": "Infer pipe not initialized in worker."})
            continue

        # Unpack task, now including wavvae_load_mode for this specific task (though worker is already configured)
        # The worker is configured at launch. Task-specific mode isn't needed here if worker is fixed.
        inp_audio_path, inp_npy_path, inp_text, infer_timestep, p_w, t_w = task
        
        print(f"Worker {device_id} received task: Audio: {inp_audio_path}, NPY: {inp_npy_path}, Text: {inp_text}")

        try:
            # This part needs to be robust if inp_audio_path is None and has_vae_encoder is False
            if inp_audio_path is None and not infer_pipe.has_vae_encoder:
                 raise ValueError("Audio input is required when VAE encoder is not available/active.")
            
            file_content = None
            if inp_audio_path:
                # Ensure .wav conversion and cutting happens, and path is correct
                # Note: convert_to_wav and cut_wav modify files in place or create new ones.
                # This might be problematic for concurrent requests if they use the same input file names.
                # For an API, it's better to work with byte streams or unique temporary files.
                # For this step, we'll keep it similar to existing logic.
                convert_to_wav(inp_audio_path) # This expects a path
                wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
                cut_wav(wav_path, max_len=28) # Max length for prompt
                with open(wav_path, 'rb') as file:
                    file_content = file.read()
            
            # Preprocess. If file_content is None, it means no audio prompt (e.g. text-only generation if model supports)
            # The current MegaTTS3DiTInfer.preprocess expects audio_bytes.
            # If inp_audio_path was None, file_content will be None.
            # This path needs careful handling if audio prompt is truly optional.
            # For now, assume audio_bytes (file_content) is required by preprocess.
            if file_content is None:
                raise ValueError("Audio prompt content is missing for preprocessing.")

            resource_context = infer_pipe.preprocess(file_content, latent_file=inp_npy_path)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_q.put({"wav_bytes": wav_bytes, "error": None})
        except Exception as e:
            print(f"!!! Error in model_worker {device_id} during processing: {e}")
            traceback.print_exc()
            output_q.put({"error": str(e), "wav_bytes": None})

# --- FastAPI App Definition ---
app = FastAPI(title="MegaTTS3 REST API", version="1.0")

# --- Pydantic Models (Request/Response Schemas) ---
class SynthesizeRequest(BaseModel):
    input_wav_path: str # Path accessible by the server where the service_api.py is running
    input_npy_path: Optional[str] = None # Optional path, also server-accessible
    input_text: str
    infer_timestep: int = 32
    p_w: float = 1.4 # Corresponds to intelligibility weight
    t_w: float = 3.0 # Corresponds to similarity weight
    # Note: wavvae_load_mode for the worker is set at service startup via ENV.
    # This model defines the data structure for a synthesis request.

# --- API Endpoints ---
@app.post("/synthesize/")
async def synthesize_speech(request: SynthesizeRequest):
    # Logic for this endpoint will be implemented in the next step.
    # For now, just a placeholder or basic response.
    # This function will eventually put a task on the input_queue
    # and get a result from output_queue.
    
    # Placeholder for now:
    # print(f"Received synthesis request: {request}")
    # return {"message": "Request received, processing not yet implemented.", "data": request.dict()}
    
    # Actual logic will involve:
    # 1. Constructing task from request
    # 2. input_queue.put(task)
    # 3. result = output_queue.get()
    global input_queue, output_queue # Ensure access to global queues

    if not input_queue or not output_queue:
        # This might happen if startup_event failed or called out of order
        raise HTTPException(status_code=503, detail="Service not ready, queues not initialized.")

    # The wavvae_load_mode is implicitly handled by how the workers were started.
    # The request data (e.g. presence of input_npy_path) will guide the model_worker.
    task = (
        request.input_wav_path,
        request.input_npy_path,
        request.input_text,
        request.infer_timestep,
        request.p_w,
        request.t_w
    )

    try:
        print(f"Putting task on queue: {task}")
        input_queue.put(task)
        
        # Wait for the result from the worker.
        # Consider adding a timeout mechanism here for production.
        result = output_queue.get(timeout=60) # Timeout after 60 seconds
        
        print(f"Got result from worker: {result.keys() if result else 'None'}")

        if result and result.get("error"):
            # If the worker reported an error
            raise HTTPException(status_code=500, detail=f"Error during synthesis: {result['error']}")
        
        if result and result.get("wav_bytes"):
            wav_bytes = result["wav_bytes"]
            # Return the WAV audio data as a streaming response
            return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
        else:
            # Should not happen if error is also None, but as a safeguard
            raise HTTPException(status_code=500, detail="Synthesis failed: No audio data and no error message from worker.")

    except mp.queues.Empty: # Specific exception for queue.get(timeout=...)
        raise HTTPException(status_code=504, detail="Synthesis request timed out waiting for worker.")
    except Exception as e:
        # Catch any other unexpected errors during queue communication or processing
        print(f"!!! Unexpected error in /synthesize endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- Startup and Shutdown Events ---
def startup_event():
    global mp_manager, input_queue, output_queue, processes
    
    print("Starting up API and model workers...")
    mp.set_start_method('spawn', force=True) # Set start method for multiprocessing
    mp_manager = mp.Manager()
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()

    # TODO: Make num_workers and default_wavvae_load_mode configurable (e.g., via env vars or CLI args to service_api.py)
    num_workers = 1 
    # Default load mode for workers. Can be overridden by API request if we implement per-request mode later.
    # For now, all workers use this mode.
    worker_wavvae_load_mode = os.environ.get("SERVICE_WAVVAE_LOAD_MODE", "full") 

    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices_list = []
    if devices_str:
        devices_list = devices_str.split(',')
    elif torch.cuda.is_available(): # If CUDA_VISIBLE_DEVICES is not set, use all available GPUs
        devices_list = [str(i) for i in range(torch.cuda.device_count())]

    if not devices_list: # No CUDA devices specified or found, default to CPU for workers
        print("No CUDA devices specified or found. Workers will use CPU.")
        # Create CPU workers
        for i in range(num_workers):
            p = mp.Process(target=model_worker_main, args=(input_queue, output_queue, None, worker_wavvae_load_mode))
            p.start()
            processes.append(p)
    else:
        # Create GPU workers
        for i in range(num_workers):
            device_id_for_worker = devices_list[i % len(devices_list)]
            p = mp.Process(target=model_worker_main, args=(input_queue, output_queue, device_id_for_worker, worker_wavvae_load_mode))
            p.start()
            processes.append(p)
    
    print(f"{num_workers} model worker(s) started.")

def shutdown_event():
    global input_queue, processes
    print("Shutting down API and model workers...")
    if input_queue:
        for _ in range(len(processes)):
            input_queue.put(None) # Send stop signal to each worker
    for p in processes:
        p.join(timeout=5) # Wait for workers to finish
        if p.is_alive():
            p.terminate() # Force terminate if still alive
    print("Model workers shut down.")

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# --- Main block to run the app (for direct execution) ---
if __name__ == "__main__":
    # Configuration for Uvicorn server
    # Consider making host, port, log_level, and workers configurable via CLI args to this script
    # For now, hardcoding standard defaults.
    port = int(os.environ.get("SERVICE_PORT", 8000))
    host = os.environ.get("SERVICE_HOST", "0.0.0.0")
    log_level = os.environ.get("SERVICE_LOG_LEVEL", "info")
    
    # Note: Uvicorn's --workers parameter is for its own HTTP worker processes,
    # not to be confused with our multiprocessing model_worker_main processes.
    # Generally, for a CPU-bound or IO-bound app, Uvicorn workers can be > 1.
    # For a ML service that has its own internal parallelism (like our mp.Process workers),
    # it's often recommended to run uvicorn with 1 worker and let the app manage its own pool,
    # or carefully tune both.
    # For simplicity, we'll use 1 uvicorn worker here.
    
    print(f"Starting Uvicorn server on {host}:{port} with log_level {log_level}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
