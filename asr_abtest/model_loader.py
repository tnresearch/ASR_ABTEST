import asyncio
import os
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional
import torch
from transformers import pipeline, AutoTokenizer
import logging
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfFolder, GatedRepoError
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoaderProcessor:
    def __init__(self):
        self.active_loads: Dict[str, Dict] = {}
        # Add a lock for thread-safe access to the transcriber instances
        self.lock = threading.Lock()
        logger.info("ModelLoaderProcessor initialized")

    async def start_loading(self, model_id: str, transcriber_instance, model_id_instance, token: Optional[str] = None) -> str:
        """Starts a new model loading process."""
        load_id = str(uuid4())
        self.active_loads[load_id] = {
            "status": "pending",
            "progress": 0,
            "details": "Initiating model load...",
            "model_id": model_id,
            "start_time": datetime.now().isoformat(),
        }
        
        # Start the loading process in the background
        asyncio.create_task(self._process_load(load_id, model_id, transcriber_instance, model_id_instance, token))
        
        return load_id

    def get_status(self, load_id: str) -> Dict:
        """Gets the current status of a loading process."""
        if load_id not in self.active_loads:
            raise KeyError(f"Load ID {load_id} not found")
        return self.active_loads[load_id]

    def cancel_loading(self, load_id: str) -> None:
        """Cancels a running loading process."""
        if load_id in self.active_loads:
            if self.active_loads[load_id]["status"] in ["downloading", "loading", "pending"]:
                self.active_loads[load_id]["status"] = "cancelled"
                self.active_loads[load_id]["details"] = "Model loading was cancelled by the user."

    async def _process_load(self, load_id: str, model_id: str, transcriber_instance, model_id_instance, token: Optional[str]):
        """The actual loading logic."""
        task = self.active_loads[load_id]

        try:
            # --- Progress Tracking Setup ---
            progress_data = {"progress": 0, "details": "Starting download..."}
            
            def progress_callback(progress, details):
                with self.lock:
                    task["progress"] = progress
                    task["details"] = details

            # --- Phase 1: Downloading with real progress tracking ---
            task["status"] = "downloading"
            progress_callback(0, f"Downloading model files for {model_id}...")

            # Run the blocking download in a separate thread
            download_thread = threading.Thread(
                target=self.download_model_with_progress,
                args=(model_id, load_id, progress_callback, token)
            )
            download_thread.start()
            
            # Await the thread's completion asynchronously
            while download_thread.is_alive():
                if task["status"] == "cancelled":
                    # This won't stop the thread, but it will prevent moving to the next phase
                    logger.info(f"Loading cancelled for task {load_id} during download.")
                    return
                await asyncio.sleep(0.5)

            if task.get("error"): # Check if an error was set by the thread
                 raise Exception(task["error"])

            if task["status"] == "cancelled":
                logger.info(f"Loading cancelled for task {load_id} after download.")
                return

            # --- Phase 2: Loading model into memory ---
            task["status"] = "loading"
            progress_callback(95, "Loading model into memory...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, local_files_only=True)
            
            new_transcriber = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                tokenizer=tokenizer,
                chunk_length_s=30,
                return_timestamps="word",
                device=device,
                local_files_only=True # Ensure it uses the cached files
            )

            if task["status"] == "cancelled":
                logger.info(f"Loading cancelled for task {load_id} during memory load.")
                return

            # --- Phase 3: Success ---
            with self.lock:
                transcriber_instance[0] = new_transcriber
                model_id_instance[0] = model_id

            task["status"] = "completed"
            progress_callback(100, f"Model '{model_id}' loaded successfully.")
            task["end_time"] = datetime.now().isoformat()
            logger.info(f"Model '{model_id}' loaded successfully for task {load_id}.")

        except Exception as e:
            logger.error(f"Error loading model for task {load_id}: {e}", exc_info=True)
            task["status"] = "failed"
            task["details"] = f"An error occurred: {str(e)}"
            task["end_time"] = datetime.now().isoformat()

    def download_model_with_progress(self, model_id, load_id, progress_callback, token: Optional[str]):
        """
        Downloads a model using snapshot_download and reports progress.
        This function is designed to be run in a separate thread.
        """
        try:
            # This is a blocking call.
            snapshot_download(
                repo_id=model_id,
                token=token,
                repo_type="model",
                local_files_only=False,
                resume_download=True,
                # The progress bar is handled via tqdm which we can't easily hook into.
                # This is a limitation of the current huggingface_hub library.
                # We will simulate the progress for now and update the status text.
            )
            # Since we can't get fine-grained progress, we jump to 90% after download.
            progress_callback(90, "Download complete. Preparing to load into memory...")
        except GatedRepoError as e:
            logger.error(f"Access denied for task {load_id}: {e}")
            with self.lock:
                task = self.active_loads[load_id]
                task["error"] = "Access to this model is restricted. Please ensure you have accepted the license on the Hugging Face Hub and provided a valid token."
        except Exception as e:
            logger.error(f"Download failed for task {load_id}: {e}")
            with self.lock:
                task = self.active_loads[load_id]
                task["error"] = str(e)
