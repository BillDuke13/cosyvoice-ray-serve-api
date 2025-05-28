"""
CosyVoice Ray Serve API

A production-ready HTTP API for the CosyVoice Text-to-Speech (TTS) model using Ray Serve.
This API provides scalable and high-performance TTS capabilities, including:
- Multi-language synthesis (Chinese, English, Japanese, Korean, Cantonese)
- Zero-shot voice cloning from a short audio sample
- Cross-lingual voice synthesis (maintaining voice identity across languages)
- Instruction-based voice control (e.g., emotion, style)
- Real-time streaming audio generation for low-latency applications

The service is designed for robustness with comprehensive error handling,
logging, and efficient resource management, including automatic GPU detection
and utilization.
"""

import asyncio
import io
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

# Configure CUDA environment for optimal performance and to avoid common issues.
# PCI_BUS_ID ensures CUDA device order matches nvidia-smi.
# CUDA_LAUNCH_BLOCKING can help with debugging CUDA errors by making them synchronous.
# PYTORCH_CUDA_ALLOC_CONF can fine-tune memory allocation.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0' # Set to '1' for debugging CUDA errors
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import ray
import torch
import torchaudio
from modelscope import snapshot_download  # For downloading pre-trained models
from ray import serve
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware  # For enabling CORS
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

# Assuming CosyVoice modules are structured as per the original project
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Type hints for clarity
PathLike = Union[str, Path]
JsonDict = Dict[str, Any]

# --- Configuration Constants ---
# Default voice prompt file (ensure this path is correct relative to asset_dir)
VOICE_PROMPTS: Dict[str, str] = {"qwen": "qwen.wav"} 
DEFAULT_VOICE: str = "qwen" # Default voice key from VOICE_PROMPTS
SAMPLE_RATE: int = 24000 # Standard sample rate for CosyVoice output
REFERENCE_AUDIO_SAMPLE_RATE: int = 16000 # Sample rate for processed reference audio

# --- Directory Setup ---
# Determine project root and other important directories
# __file__ refers to the current script (api.py)
ROOT_DIR: Path = Path(__file__).resolve().parent.absolute()
TMP_DIR: Path = ROOT_DIR / 'tmp'
LOGS_DIR: Path = ROOT_DIR / 'logs'
ASSET_DIR: Path = ROOT_DIR / 'asset'
MODEL_DIR: Path = ROOT_DIR / 'pretrained_models' / 'CosyVoice2-0.5B'

# Create necessary directories if they don't exist
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR.parent, exist_ok=True) # Ensure pretrained_models directory exists

# --- Logging Configuration ---
# Set up comprehensive logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to standard output
)

log_file_path: Path = LOGS_DIR / 'cosyvoice_api.log'
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler) # Add file handler to root logger
logger = logging.getLogger(__name__) # Get a logger for this specific module

# --- GPU and PyTorch Configuration ---
# Log PyTorch and CUDA details if CUDA is available
if torch.cuda.is_available():
    # It's good practice to initialize CUDA explicitly if you're managing devices.
    # However, PyTorch usually handles this. If issues arise, uncomment:
    # torch.cuda.init() 
    torch.cuda.empty_cache() # Clear any cached memory

    # Enable features for better performance on compatible hardware (e.g., Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True # Good for when input sizes don't vary much
    torch.backends.cudnn.deterministic = False # Set to True if reproducibility is critical

    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"CUDA Device {i}: {props.name} (Total Memory: {props.total_memory / 1024**3:.1f}GB)")

    if torch.cuda.device_count() > 0:
        # Ray Serve handles GPU assignment, so setting a default device here
        # might be overridden or conflict. It's generally safer to let Ray manage it.
        # torch.cuda.set_device(0) 
        # logger.info(f"Default CUDA Device (globally set): {torch.cuda.get_device_name(0)}")
        pass
else:
    logger.warning("CUDA is not available. The service will run on CPU only.")

# --- Environment and Path Setup ---
# Add project directories to PATH and PYTHONPATH if necessary for dependencies
# This is often needed for custom C++ extensions or when projects have complex structures.
# Example: os.environ['PATH'] = f"{ROOT_DIR}/bin:{os.environ['PATH']}"
# Example: sys.path.append(str(ROOT_DIR / 'third_party' / 'some_module'))

# --- Utility Class for Audio Processing ---
class AudioProcessor:
    """
    Handles audio processing tasks like format conversion, normalization, and validation.
    """
    @staticmethod
    def process_reference_audio(
        input_audio_path: PathLike,
        output_wav_path: PathLike,
        target_sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE
    ) -> None:
        """
        Converts an input audio file to a mono WAV file with a specific sample rate
        using FFmpeg. This is typically used for preparing reference audio.

        Args:
            input_audio_path: Path to the input audio file.
            output_wav_path: Path to save the processed WAV file.
            target_sample_rate: The desired sample rate for the output.
        
        Raises:
            FileNotFoundError: If the input audio file does not exist.
            subprocess.CalledProcessError: If FFmpeg command fails.
        """
        if not Path(input_audio_path).exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        logger.info(f"Processing reference audio: {input_audio_path} to {output_wav_path} at {target_sample_rate}Hz")
        try:
            # FFmpeg command:
            # -y: Overwrite output file if it exists
            # -i: Input file
            # -ar: Audio sample rate
            # -ac: Audio channels (1 for mono)
            # -acodec: Audio codec (pcm_s16le for WAV)
            # -hide_banner: Suppress FFmpeg startup banner
            # -loglevel error: Show only errors
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(input_audio_path),
                    "-ar", str(target_sample_rate),
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(output_wav_path),
                    "-hide_banner", "-loglevel", "error"
                ],
                check=True, # Raise an exception for non-zero exit codes
                capture_output=True, # Capture stdout and stderr
                text=True # Decode stdout/stderr as text
            )
            logger.info(f"Successfully processed reference audio: {output_wav_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg processing failed for {input_audio_path}. Error: {e.stderr}")
            raise
        except Exception as e_gen:
            logger.error(f"An unexpected error occurred during FFmpeg processing: {e_gen}")
            raise


    @staticmethod
    def normalize_audio_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes an audio tensor to the range [-1, 1].

        Args:
            audio_tensor: A PyTorch tensor representing audio data.

        Returns:
            The normalized audio tensor.
        """
        if not isinstance(audio_tensor, torch.Tensor):
            logger.warning("Attempted to normalize non-tensor data.")
            return audio_tensor # Or raise error

        # Ensure tensor is float32 for processing
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()
        
        # Handle potential multi-channel audio (e.g., stereo) by normalizing each channel
        if audio_tensor.dim() > 1: # e.g., [channels, samples] or [batch, channels, samples]
            max_vals = audio_tensor.abs().max(dim=-1, keepdim=True)[0]
            # Avoid division by zero for silent channels
            audio_tensor = torch.where(max_vals > 1e-6, audio_tensor / max_vals, audio_tensor)
        else: # Single channel audio [samples]
            max_val = audio_tensor.abs().max()
            if max_val > 1e-6: # Avoid division by zero for silence
                audio_tensor = audio_tensor / max_val
        
        return audio_tensor

    @staticmethod
    def validate_audio_tensor(audio_tensor: torch.Tensor) -> bool:
        """
        Validates an audio tensor to ensure it's suitable for further processing or output.
        Checks for NaN, Inf, and empty tensors.

        Args:
            audio_tensor: The audio tensor to validate.

        Returns:
            True if the tensor is valid, False otherwise.
        """
        if not isinstance(audio_tensor, torch.Tensor):
            logger.warning("Validation failed: Input is not a PyTorch tensor.")
            return False
        
        if audio_tensor.numel() == 0: # Check if tensor is empty
            logger.warning("Validation failed: Audio tensor is empty.")
            return False
            
        if torch.isnan(audio_tensor).any(): # Check for Not-a-Number values
            logger.warning("Validation failed: Audio tensor contains NaN values.")
            return False
            
        if torch.isinf(audio_tensor).any(): # Check for infinity values
            logger.warning("Validation failed: Audio tensor contains Inf values.")
            return False
            
        return True

    @staticmethod
    async def save_audio_to_mp3(
        audio_tensor: torch.Tensor,
        output_mp3_path: PathLike,
        sample_rate: int = SAMPLE_RATE
    ) -> PathLike:
        """
        Saves an audio tensor to an MP3 file using FFmpeg for encoding.
        A temporary WAV file is created first, then converted to MP3.

        Args:
            audio_tensor: The audio data as a PyTorch tensor.
            output_mp3_path: The path to save the final MP3 file.
            sample_rate: The sample rate of the audio tensor.

        Returns:
            The path to the saved MP3 file.

        Raises:
            Exception: If audio saving or conversion fails.
        """
        # Create a unique temporary WAV file path
        timestamp = int(time.time_ns() // 1000) # Microseconds for uniqueness
        temp_wav_path = TMP_DIR / f"temp_audio_{timestamp}.wav"
        
        try:
            # Ensure audio tensor is on CPU and in the correct format for torchaudio.save
            # Expected shape: [channels, samples] or [samples]
            if audio_tensor.is_cuda:
                audio_tensor = audio_tensor.cpu()
            if audio_tensor.dim() == 1: # Add channel dimension if it's mono [samples] -> [1, samples]
                audio_tensor = audio_tensor.unsqueeze(0)

            torchaudio.save(
                str(temp_wav_path),
                audio_tensor,
                sample_rate,
                encoding='PCM_S', # Signed 16-bit PCM
                bits_per_sample=16
            )
            logger.info(f"Temporary WAV file saved: {temp_wav_path}")
            
            # Convert WAV to MP3 using FFmpeg
            # -i: Input file (temp WAV)
            # -codec:a libmp3lame: Use LAME MP3 encoder
            # -qscale:a 2: Variable bitrate encoding, quality level 2 (good quality)
            # -y: Overwrite output
            # -hide_banner -loglevel error: Suppress verbose output
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_wav_path),
                "-codec:a", "libmp3lame",
                "-qscale:a", "2", # VBR quality, 0-9 (lower is better)
                str(output_mp3_path),
                "-hide_banner", "-loglevel", "error"
            ], check=True, capture_output=True)
            
            if not Path(output_mp3_path).exists() or Path(output_mp3_path).stat().st_size == 0:
                raise Exception(f"MP3 file failed to save or is empty: {output_mp3_path}")
            
            file_size_kb = Path(output_mp3_path).stat().st_size / 1024
            logger.info(f"Generated MP3 file: {output_mp3_path} (Size: {file_size_kb:.1f}KB)")
            
            # Basic sanity check for very small files (e.g., less than 1KB might indicate an issue)
            if file_size_kb < 0.5: 
                logger.warning(f"Generated MP3 file {output_mp3_path} is very small ({file_size_kb:.1f}KB). This might indicate an issue.")
            
            return output_mp3_path
        except torchaudio.TorchaudioException as e_torch:
            logger.error(f"Torchaudio failed to save temporary WAV file: {e_torch}")
            raise Exception("Audio saving failed (torchaudio error).") from e_torch
        except subprocess.CalledProcessError as e_ffmpeg:
            logger.error(f"FFmpeg MP3 conversion failed. Error: {e_ffmpeg.stderr}")
            raise Exception("Audio conversion to MP3 failed (ffmpeg error).") from e_ffmpeg
        except Exception as e_gen:
            logger.error(f"An unexpected error occurred during audio saving/conversion: {e_gen}")
            raise
        finally:
            # Clean up the temporary WAV file
            if temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                    logger.debug(f"Deleted temporary WAV file: {temp_wav_path}")
                except OSError as e_os:
                    logger.warning(f"Failed to delete temporary WAV file {temp_wav_path}: {e_os}")


# --- Ray Serve Deployment Configuration ---
# Defines how the CosyVoiceService is deployed and managed by Ray Serve.
# Includes settings for autoscaling, health checks, and resource allocation.
@serve.deployment(
    name="CosyVoiceAPI", # Descriptive name for the deployment
    # Autoscaling configuration:
    # Adjust these based on expected load and available resources.
    autoscaling_config={
        "min_replicas": int(os.environ.get("MIN_REPLICAS", "1")),
        "initial_replicas": int(os.environ.get("INITIAL_REPLICAS", "1")),
        "max_replicas": int(os.environ.get("MAX_REPLICAS", "2")), # Max number of model server replicas
        "target_ongoing_requests": int(os.environ.get("TARGET_ONGOING_REQUESTS", "2")), # Target concurrent requests per replica
        "metrics_interval_s": 10.0, # How often to scrape metrics
        "look_back_period_s": 30.0, # Time window for averaging metrics
        "smoothing_factor": 1.0, # Controls responsiveness of autoscaling (1.0 = no smoothing)
        # "downscale_delay_s": 600.0, # Delay before scaling down
        # "upscale_delay_s": 30.0,    # Delay before scaling up
    },
    # Resource allocation per replica:
    # Request GPUs if available and desired. Ray will manage allocation.
    # num_gpus=1 if torch.cuda.is_available() else 0, # Request 1 GPU per replica if CUDA is available
    ray_actor_options={"num_gpus": 1} if torch.cuda.is_available() and int(os.environ.get("NUM_GPUS_PER_REPLICA", "1")) > 0 else {},

    # Health check configuration:
    health_check_period_s=int(os.environ.get("HEALTH_CHECK_PERIOD_S", "15")), # Frequency of health checks
    health_check_timeout_s=int(os.environ.get("HEALTH_CHECK_TIMEOUT_S", "30")), # Timeout for health check response

    # Request handling:
    max_ongoing_requests=int(os.environ.get("MAX_ONGOING_REQUESTS_PER_REPLICA", "5")), # Max concurrent requests a single replica handles
    max_queued_requests=int(os.environ.get("MAX_QUEUED_REQUESTS_DEPLOYMENT", "20")), # Max requests to queue at the deployment level

    # Graceful shutdown:
    graceful_shutdown_timeout_s=int(os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT_S", "60")), # Time for replicas to finish requests before shutdown
    graceful_shutdown_wait_loop_s=int(os.environ.get("GRACEFUL_SHUTDOWN_WAIT_LOOP_S", "5")), # Interval to check for request completion during shutdown
    
    # User-defined configuration accessible within the service
    user_config={
        "default_voice": DEFAULT_VOICE,
        "sample_rate": SAMPLE_RATE,
        "model_dir_name": MODEL_DIR.name, # Pass only the directory name
    }
)
class CosyVoiceService:
    """
    Ray Serve service class for CosyVoice TTS.
    Manages model loading, inference, and request handling.
    """
    def __init__(self, user_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the CosyVoice service.
        This method is called once per replica when it's created.
        It handles device setup, model loading, and other one-time initializations.
        """
        self.user_config = user_config or {}
        self._reference_audio_cache: Dict[str, PathLike] = {} # Cache for processed reference audio paths
        
        self.device: str = self._setup_device() # Determine 'cuda:X' or 'cpu'
        self.model: Optional[CosyVoice2] = None # Placeholder for the TTS model
        self.audio_processor = AudioProcessor() # Instantiate audio utility class
        
        # Load voice prompts from ASSET_DIR
        self.voice_prompts: Dict[str, Path] = {
            key: ASSET_DIR / Path(filename).name # Ensure only filename is used
            for key, filename in VOICE_PROMPTS.items()
        }
        self.default_voice: str = self.user_config.get("default_voice", DEFAULT_VOICE)

        # Ensure all specified voice prompt files exist
        for voice_key, prompt_path in self.voice_prompts.items():
            if not prompt_path.exists():
                logger.error(f"Voice prompt file for '{voice_key}' not found at {prompt_path}. This voice will be unavailable.")
                # Optionally, remove this voice from available prompts or raise an error
            else:
                logger.info(f"Voice prompt '{voice_key}' loaded from {prompt_path}")

        if not self.voice_prompts:
             logger.warning(f"No voice prompts found in {ASSET_DIR}. TTS with default prompts might fail.")
        elif self.default_voice not in self.voice_prompts:
            logger.warning(f"Default voice '{self.default_voice}' not found in available prompts. TTS may fail if no voice is specified.")


        self._download_and_setup_models() # Download (if needed) and load the model

        # Register a cleanup handler for when the actor exits
        # This is useful for releasing resources like GPU memory.
        # Note: Ray actor lifecycle management might also handle some of this.
        try:
            import atexit  # Standard library module for registering exit functions
            atexit.register(self._cleanup_resources)
            logger.info("Registered resource cleanup handler.")
        except Exception as e:
            logger.error(f"Failed to register cleanup handler: {e}")

    def _setup_device(self) -> str:
        """
        Sets up the computation device (GPU or CPU) for this replica.
        Ray Serve assigns GPUs if `num_gpus` is set in @serve.deployment.
        This method determines which specific GPU (if any) is assigned to this actor/replica.

        Returns:
            A string representing the PyTorch device (e.g., "cuda:0", "cpu").
        """
        if torch.cuda.is_available():
            # Ray provides assigned GPU IDs to the actor.
            # This is the most reliable way to get the correct GPU in a Ray cluster.
            try:
                gpu_ids = ray.get_gpu_ids() # Returns a list of GPU IDs assigned to this actor
                if gpu_ids:
                    # Typically, one GPU is assigned per actor if num_gpus=1
                    assigned_gpu_id = int(gpu_ids[0]) 
                    device_str = f"cuda:{assigned_gpu_id}"
                    
                    # Verify the device by trying to use it
                    # This also helps "warm up" the CUDA context for this GPU.
                    _ = torch.tensor([1.0]).to(device_str) 
                    
                    logger.info(f"Replica assigned to GPU: {assigned_gpu_id}. Using device: {device_str}")
                    # Log memory details for this specific GPU
                    # torch.cuda.get_device_properties requires an int or torch.device
                    props = torch.cuda.get_device_properties(assigned_gpu_id)
                    logger.info(f"Properties for {device_str}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
                    logger.info(f"Initial memory allocated on {device_str}: {torch.cuda.memory_allocated(device_str) / 1024**2:.1f}MB")
                    return device_str
                else:
                    logger.warning("CUDA is available, but no GPUs were assigned to this replica by Ray. Falling back to CPU.")
                    return "cpu"
            except Exception as e:
                logger.error(f"Error detecting or setting up assigned GPU: {e}. Falling back to CPU.")
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Attempt to clear cache if error occurred
                return "cpu"
        else:
            logger.info("CUDA not available. This replica will use CPU.")
            return "cpu"

    def _cleanup_resources(self) -> None:
        """
        Cleans up resources when the replica is shutting down.
        This is particularly important for releasing GPU memory.
        """
        logger.info(f"Cleaning up resources for replica on device {self.device}...")
        try:
            if hasattr(self, 'model') and self.model is not None:
                # If the model has a specific cleanup method, call it.
                # e.g., if self.model.cleanup(): self.model.cleanup()
                del self.model # Remove reference to allow garbage collection
                self.model = None
                logger.info("CosyVoice model deleted.")

            if self.device.startswith("cuda") and torch.cuda.is_available():
                try:
                    # Clear PyTorch's CUDA memory cache for the assigned device.
                    # This helps ensure memory is freed for other processes or replicas.
                    torch.cuda.empty_cache() 
                    logger.info(f"Cleared PyTorch CUDA memory cache for {self.device}.")
                except Exception as e:
                    logger.error(f"Error clearing CUDA memory cache on {self.device}: {e}")
            
            # Clean up temporary files in the cache, if any
            for path_to_delete in self._reference_audio_cache.values():
                try:
                    if Path(path_to_delete).exists():
                        os.unlink(path_to_delete)
                        logger.info(f"Deleted cached processed audio: {path_to_delete}")
                except OSError as e_os:
                    logger.warning(f"Error deleting cached file {path_to_delete}: {e_os}")
            self._reference_audio_cache.clear()

            logger.info("Resource cleanup complete.")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}", exc_info=True)
    
    def _download_and_setup_models(self) -> None:
        """
        Downloads the CosyVoice pre-trained models from ModelScope if they don't exist locally,
        then loads the model onto the configured device.
        """
        model_id = 'iic/CosyVoice2-0.5B' # ModelScope model identifier
        # Use MODEL_DIR directly as it's already an absolute path
        local_model_path = MODEL_DIR 

        try:
            # Check if the model directory exists and seems valid (e.g., contains expected files)
            # A more robust check might involve looking for specific model files.
            if not local_model_path.exists() or not any(local_model_path.iterdir()):
                logger.info(f"CosyVoice model not found at {local_model_path}. Downloading from ModelScope ({model_id})...")
                # snapshot_download will create local_model_path if it doesn't exist.
                snapshot_download(
                    model_id,
                    local_dir=str(local_model_path), # Must be a string
                    revision='main' # Or a specific commit hash
                )
                logger.info(f"Model download completed to {local_model_path}.")
            else:
                logger.info(f"CosyVoice model found at {local_model_path}. Skipping download.")

            # Load the CosyVoice2 model
            logger.info(f"Initializing CosyVoice2 model on device: {self.device} from path: {local_model_path}")
            
            use_cuda_features = self.device.startswith('cuda')
            
            # CosyVoice2 expects the parent directory of the actual model files
            # e.g., if models are in 'pretrained_models/CosyVoice2-0.5B', pass this path.
            self.model = CosyVoice2(
                model_dir=str(local_model_path), # Path to the model directory
                # load_jit: Use JIT compiled model (often faster on GPU)
                load_jit=use_cuda_features, 
                # load_trt: Use TensorRT compiled model (can be even faster, but requires TensorRT setup)
                load_trt=False, # Set to True if TensorRT is available and desired
                # fp16: Use half-precision floating point (reduces memory, can speed up on compatible GPUs)
                fp16=use_cuda_features 
            )
            logger.info(f"Successfully initialized CosyVoice2 model on {self.device}.")

            if use_cuda_features and torch.cuda.is_available():
                # Log memory usage after model loading
                allocated_mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                reserved_mem_mb = torch.cuda.memory_reserved(self.device) / 1024**2
                logger.info(f"GPU memory after model init on {self.device}: Allocated={allocated_mem_mb:.1f}MB, Reserved={reserved_mem_mb:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to download or initialize CosyVoice model: {e}", exc_info=True)
            # If GPU initialization failed, attempt to fall back to CPU if not already on CPU.
            if self.device.startswith("cuda"):
                logger.warning("Model initialization on GPU failed. Attempting to fall back to CPU.")
                self.device = "cpu" # Switch device to CPU
                try:
                    self.model = CosyVoice2(
                        model_dir=str(local_model_path),
                        load_jit=False, # JIT typically not beneficial for CPU
                        load_trt=False,
                        fp16=False # FP16 usually not for CPU
                    )
                    logger.info("Successfully initialized CosyVoice2 model on CPU after GPU fallback.")
                except Exception as e_cpu:
                    logger.error(f"Failed to initialize CosyVoice model on CPU after fallback: {e_cpu}", exc_info=True)
                    raise RuntimeError("CosyVoice model could not be loaded on GPU or CPU.") from e_cpu
            else:
                # If already on CPU or some other error, re-raise
                raise RuntimeError(f"CosyVoice model could not be loaded: {e}") from e

    async def _get_processed_reference_audio(self, input_audio_path: PathLike) -> PathLike:
        """
        Processes a reference audio file (e.g., converts to WAV, resamples) and caches the result.
        This avoids re-processing the same file multiple times.

        Args:
            input_audio_path: Path to the raw reference audio file.

        Returns:
            Path to the processed (and cached) reference audio file.
        
        Raises:
            FileNotFoundError: If the input audio file cannot be found.
            Exception: If audio processing fails.
        """
        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {input_path}")

        # Create a cache key based on file path and modification time to detect changes
        try:
            mod_time = input_path.stat().st_mtime
        except OSError: # Handle potential race condition if file is deleted
            mod_time = time.time() 
        cache_key = f"{str(input_path)}_{mod_time}"
        
        if cache_key in self._reference_audio_cache:
            cached_path = self._reference_audio_cache[cache_key]
            if Path(cached_path).exists(): # Ensure cached file still exists
                logger.info(f"Using cached processed reference audio: {cached_path}")
                return cached_path
            else:
                logger.warning(f"Cached reference audio {cached_path} not found. Re-processing.")
                del self._reference_audio_cache[cache_key] # Remove stale entry

        # If not in cache or stale, process the audio
        # Create a unique name for the processed file in TMP_DIR
        timestamp = int(time.time_ns() // 1000)
        processed_filename = f"processed_ref_{input_path.stem}_{timestamp}.wav"
        processed_output_path = TMP_DIR / processed_filename
        
        logger.info(f"Processing reference audio {input_path} for zero-shot/cross-lingual TTS...")
        try:
            self.audio_processor.process_reference_audio(
                input_audio_path=input_path,
                output_wav_path=processed_output_path,
                target_sample_rate=REFERENCE_AUDIO_SAMPLE_RATE
            )
            self._reference_audio_cache[cache_key] = processed_output_path # Add to cache
            logger.info(f"Reference audio processed and cached: {processed_output_path}")
            return processed_output_path
        except Exception as e:
            logger.error(f"Failed to process reference audio {input_path}: {e}", exc_info=True)
            # Ensure partially created file is removed on error
            if processed_output_path.exists():
                try: os.unlink(processed_output_path)
                except OSError: pass
            raise # Re-raise the exception to be handled by the caller

    async def _generate_audio_common(
        self,
        inference_method_name: str,
        text: str,
        stream: bool,
        **kwargs
    ) -> Union[torch.Tensor, AsyncGenerator[bytes, None]]:
        """
        Common logic for generating audio using different CosyVoice inference methods.
        Handles streaming and non-streaming output.

        Args:
            inference_method_name: Name of the CosyVoice model's inference method to call
                                   (e.g., 'inference_sft', 'inference_zero_shot').
            text: The input text to synthesize.
            stream: Whether to stream the audio output.
            **kwargs: Additional arguments for the specific inference method.

        Returns:
            If stream is False, returns a concatenated torch.Tensor of the full audio.
            If stream is True, returns an AsyncGenerator yielding bytes of MP3 audio chunks.
        
        Raises:
            AttributeError: If the model or inference method is not found.
            Exception: If TTS generation fails or produces no valid audio.
        """
        if self.model is None:
            raise RuntimeError("CosyVoice model is not loaded.")
        
        inference_fn = getattr(self.model, inference_method_name, None)
        if not callable(inference_fn):
            raise AttributeError(f"Inference method '{inference_method_name}' not found in CosyVoice model.")

        logger.info(f"Starting TTS generation with method '{inference_method_name}' for text: \"{text[:50]}...\" (Stream: {stream})")
        
        # Prepare arguments for the inference function
        # The 'stream' argument for CosyVoice model methods controls if it yields chunks.
        inference_args = {**kwargs, "stream": True} # Always use model's stream for chunking

        # --- Streaming Logic ---
        if stream: 
            async def stream_generator():
                temp_files_to_clean = []
                try:
                    chunk_index = 0
                    for result_chunk in inference_fn(text, **inference_args):
                        if 'tts_speech' in result_chunk:
                            audio_tensor_chunk = result_chunk['tts_speech']
                            
                            if not self.audio_processor.validate_audio_tensor(audio_tensor_chunk):
                                logger.warning(f"Stream chunk {chunk_index}: Invalid audio tensor received, skipping.")
                                continue
                            
                            # Normalize audio chunk
                            audio_tensor_chunk = self.audio_processor.normalize_audio_tensor(audio_tensor_chunk)
                            
                            # Save chunk to a temporary MP3 file
                            timestamp = int(time.time_ns() // 1000)
                            temp_mp3_chunk_path = TMP_DIR / f"stream_chunk_{timestamp}_{chunk_index}.mp3"
                            temp_files_to_clean.append(temp_mp3_chunk_path)

                            await self.audio_processor.save_audio_to_mp3(
                                audio_tensor_chunk, temp_mp3_chunk_path, sample_rate=SAMPLE_RATE
                            )
                            
                            # Yield the content of the MP3 chunk file
                            with open(temp_mp3_chunk_path, "rb") as f_chunk:
                                yield f_chunk.read()
                            
                            logger.debug(f"Streamed MP3 chunk {chunk_index} ({temp_mp3_chunk_path.stat().st_size} bytes)")
                            chunk_index += 1
                    
                    if chunk_index == 0:
                        logger.warning(f"TTS generation for method '{inference_method_name}' produced no audio chunks.")
                        # Optionally, yield a small silent MP3 or raise an error for the client
                        # For now, it will just end the stream.
                except Exception as e_stream:
                    logger.error(f"Error during streaming TTS generation ({inference_method_name}): {e_stream}", exc_info=True)
                    # If an error occurs, the client connection might be broken.
                    # Consider how to signal this error if possible (e.g., special last chunk - tricky with HTTP).
                finally:
                    # Clean up all temporary chunk files
                    for temp_file in temp_files_to_clean:
                        if temp_file.exists():
                            try: os.unlink(temp_file)
                            except OSError: logger.warning(f"Failed to delete temp stream chunk: {temp_file}")
                    logger.info(f"Finished streaming for '{inference_method_name}'. Cleaned {len(temp_files_to_clean)} temp files.")
            
            return stream_generator() # Return the async generator for streaming response

        # --- Non-Streaming Logic ---
        else:
            collected_audio_tensors: List[torch.Tensor] = []
            chunk_index = 0
            for result_chunk in inference_fn(text, **inference_args): # Still use model's stream=True
                if 'tts_speech' in result_chunk:
                    audio_tensor_chunk = result_chunk['tts_speech']
                    if not self.audio_processor.validate_audio_tensor(audio_tensor_chunk):
                        logger.warning(f"Non-stream chunk {chunk_index}: Invalid audio tensor, skipping.")
                        continue
                    
                    audio_tensor_chunk = self.audio_processor.normalize_audio_tensor(audio_tensor_chunk)
                    collected_audio_tensors.append(audio_tensor_chunk.cpu()) # Move to CPU before collecting
                    chunk_index +=1
            
            if not collected_audio_tensors:
                logger.error(f"TTS generation for method '{inference_method_name}' produced no valid audio output.")
                raise Exception("TTS generation failed to produce audio.")
            
            # Concatenate all audio chunks into a single tensor
            # Ensure they are all on the same device (CPU) and have compatible shapes
            # CosyVoice usually outputs [1, num_samples] or [num_samples]
            # If shapes are [1, N], [1, M], etc., cat along dim 1.
            # If shapes are [N], [M], etc., cat along dim 0.
            # Assuming chunks are [num_samples] or [1, num_samples] and we want to cat along sample dimension.
            # If [1, N], then cat dim=-1 (or dim=1). If [N], then cat dim=-1 (or dim=0).
            # CosyVoice seems to output tensors that can be directly concatenated along the last dimension.
            try:
                full_audio_tensor = torch.cat(collected_audio_tensors, dim=-1)
                logger.info(f"Concatenated {len(collected_audio_tensors)} audio chunks for '{inference_method_name}'. Final duration: {full_audio_tensor.shape[-1]/SAMPLE_RATE:.2f}s")
                return full_audio_tensor
            except Exception as e_cat:
                logger.error(f"Error concatenating audio chunks for '{inference_method_name}': {e_cat}", exc_info=True)
                # Log shapes for debugging
                for i, t in enumerate(collected_audio_tensors):
                    logger.error(f"Chunk {i} shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
                raise Exception("Failed to combine audio chunks.") from e_cat


    # --- API Endpoint Handlers ---

    async def health_check(self, request: Request) -> JSONResponse:
        """
        Basic health check endpoint.
        Verifies that the model is loaded and the service is responsive.
        """
        logger.info("Health check requested.")
        if self.model is None:
            logger.error("Health check failed: Model is not loaded.")
            return JSONResponse({"status": "unhealthy", "reason": "Model not loaded"}, status_code=503)
        
        # Optionally, perform a quick inference test if feasible and not too resource-intensive.
        # For now, just checking model presence is sufficient.
        return JSONResponse({
            "status": "healthy", 
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device": self.device,
            "model_loaded": self.model is not None,
            "ray_actor_id": ray.get_runtime_context().actor_id.hex() if ray.is_initialized() else "N/A",
            "ray_node_id": ray.get_runtime_context().node_id.hex() if ray.is_initialized() else "N/A",
        })

    async def get_config(self, request: Request) -> JSONResponse:
        """
        Returns the current configuration of the service.
        """
        logger.info("Configuration request received.")
        return JSONResponse({
            "default_voice": self.default_voice,
            "available_voice_prompts": list(self.voice_prompts.keys()),
            "sample_rate": SAMPLE_RATE,
            "reference_audio_sample_rate": REFERENCE_AUDIO_SAMPLE_RATE,
            "device_in_use": self.device,
            "model_directory": str(MODEL_DIR), # Expose where models are expected
            "max_replicas_config": self.user_config.get("max_replicas", os.environ.get("MAX_REPLICAS", "2")), # Example of exposing deployment config
        })

    async def tts_standard(self, request: Request) -> Response:
        """
        Handles standard Text-to-Speech requests.
        Input: JSON with "text", optionally "voice_type", "speed", "stream".
        Output: MP3 audio file or streaming audio.
        """
        try:
            payload = await request.json()
        except Exception:
            logger.warning("TTS Standard: Invalid JSON payload received.")
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")

        text = payload.get("text")
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'text' field in payload.")

        voice_type = payload.get("voice_type", self.default_voice)
        # speed = float(payload.get("speed", 1.0)) # CosyVoice SFT doesn't directly use speed param in this way
        stream = bool(payload.get("stream", False))

        if voice_type not in self.voice_prompts:
            logger.warning(f"TTS Standard: Voice type '{voice_type}' not found. Using default '{self.default_voice}'.")
            voice_type = self.default_voice # Fallback to default if specified voice is invalid
            if voice_type not in self.voice_prompts: # If default is also missing
                 raise HTTPException(status_code=400, detail=f"Default voice prompt '{self.default_voice}' is not available.")


        # The 'inference_sft' method in CosyVoice2 is typically used for standard TTS
        # It expects text and the speaker's name (voice_type).
        # The actual voice prompt audio is handled internally by CosyVoice2 based on the speaker name.
        try:
            audio_output = await self._generate_audio_common(
                inference_method_name='inference_sft',
                text=text,
                # speaker=voice_type, # CosyVoice2 `inference_sft` takes `sft_dropdown` which is the voice_type
                sft_dropdown=voice_type, # This matches the parameter name in CosyVoice2
                stream=stream
            )

            if stream: # Output is an AsyncGenerator
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else: # Output is a torch.Tensor
                timestamp = int(time.time_ns() // 1000)
                output_mp3_path = TMP_DIR / f"tts_standard_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(audio_output, output_mp3_path, SAMPLE_RATE)
                
                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                
                # Clean up the generated MP3 file
                if output_mp3_path.exists():
                    try: os.unlink(output_mp3_path)
                    except OSError: logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
                
                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException: # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            logger.error(f"TTS Standard request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}")


    async def tts_zero_shot(self, request: Request) -> Response:
        """
        Handles Zero-Shot Text-to-Speech requests for voice cloning.
        Input: FormData with "text", "reference_text", "reference_audio" (file),
               optionally "speed", "stream".
        Output: MP3 audio file or streaming audio.
        """
        try:
            form_data = await request.form()
            text = form_data.get("text")
            reference_text = form_data.get("reference_text") # Transcript of the reference audio
            reference_audio_file: Optional[UploadFile] = form_data.get("reference_audio")
            # speed = float(form_data.get("speed", 1.0)) # Speed not directly used by CosyVoice zero_shot
            stream = str(form_data.get("stream", "false")).lower() == "true"

            if not all([text, reference_text, reference_audio_file]):
                missing_fields = [
                    f for f, v in {"text": text, "reference_text": reference_text, "reference_audio": reference_audio_file}.items() if not v
                ]
                raise HTTPException(status_code=400, detail=f"Missing required form fields: {', '.join(missing_fields)}")
            
            if not isinstance(text, str) or not isinstance(reference_text, str):
                 raise HTTPException(status_code=400, detail="Fields 'text' and 'reference_text' must be strings.")

            # Save uploaded reference audio to a temporary file
            timestamp = int(time.time_ns() // 1000)
            temp_ref_audio_path = TMP_DIR / f"ref_audio_upload_{timestamp}_{reference_audio_file.filename}"
            
            try:
                with open(temp_ref_audio_path, "wb") as f_ref:
                    contents = await reference_audio_file.read()
                    f_ref.write(contents)
                logger.info(f"Zero-shot: Reference audio saved to temporary file: {temp_ref_audio_path}")

                # Process the reference audio (convert to WAV, resample) and get the path to the processed file
                processed_ref_audio_path = await self._get_processed_reference_audio(temp_ref_audio_path)
                
                # Load the processed reference audio as a tensor for CosyVoice
                # CosyVoice's `inference_zero_shot` expects `prompt_speech_16k` as a loaded tensor.
                prompt_speech_16k = load_wav(str(processed_ref_audio_path), REFERENCE_AUDIO_SAMPLE_RATE)

            finally:
                # Clean up the initially uploaded temporary reference audio file
                if temp_ref_audio_path.exists():
                    try: os.unlink(temp_ref_audio_path)
                    except OSError: logger.warning(f"Failed to delete uploaded temp ref audio: {temp_ref_audio_path}")
                # Note: The *processed* reference audio file from `_get_processed_reference_audio`
                # is managed by the cache and its cleanup logic.

            audio_output = await self._generate_audio_common(
                inference_method_name='inference_zero_shot',
                text=text,
                # Parameters for CosyVoice2 `inference_zero_shot`
                zero_shot_text=reference_text, # Transcript of the prompt
                prompt_speech_16k=prompt_speech_16k, # Loaded audio tensor of the prompt
                stream=stream
            )

            if stream:
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else:
                output_mp3_path = TMP_DIR / f"tts_zeroshot_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(audio_output, output_mp3_path, SAMPLE_RATE)
                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                if output_mp3_path.exists():
                    try: os.unlink(output_mp3_path)
                    except OSError: logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except FileNotFoundError as e_fnf: # Specifically for reference audio not found by _get_processed_reference_audio
            logger.error(f"Zero-shot TTS failed: Reference audio processing error - {e_fnf}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error with reference audio: {str(e_fnf)}")
        except Exception as e:
            logger.error(f"Zero-shot TTS request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Zero-shot TTS generation error: {str(e)}")


    async def tts_cross_lingual(self, request: Request) -> Response:
        """
        Handles Cross-Lingual Text-to-Speech requests.
        Input: FormData with "text", "reference_audio" (file), optionally "speed", "stream".
        Output: MP3 audio file or streaming audio.
        """
        try:
            form_data = await request.form()
            text = form_data.get("text")
            reference_audio_file: Optional[UploadFile] = form_data.get("reference_audio")
            # speed = float(form_data.get("speed", 1.0)) # Speed not directly used
            stream = str(form_data.get("stream", "false")).lower() == "true"

            if not text or not reference_audio_file:
                missing_fields = [f for f,v in {"text":text, "reference_audio":reference_audio_file}.items() if not v]
                raise HTTPException(status_code=400, detail=f"Missing required form fields: {', '.join(missing_fields)}")
            
            if not isinstance(text, str):
                 raise HTTPException(status_code=400, detail="Field 'text' must be a string.")

            timestamp = int(time.time_ns() // 1000)
            temp_ref_audio_path = TMP_DIR / f"ref_audio_crosslingual_{timestamp}_{reference_audio_file.filename}"
            
            try:
                with open(temp_ref_audio_path, "wb") as f_ref:
                    contents = await reference_audio_file.read()
                    f_ref.write(contents)
                logger.info(f"Cross-lingual: Reference audio saved to: {temp_ref_audio_path}")

                processed_ref_audio_path = await self._get_processed_reference_audio(temp_ref_audio_path)
                prompt_speech_16k = load_wav(str(processed_ref_audio_path), REFERENCE_AUDIO_SAMPLE_RATE)
            finally:
                if temp_ref_audio_path.exists():
                    try: os.unlink(temp_ref_audio_path)
                    except OSError: logger.warning(f"Failed to delete uploaded temp ref audio: {temp_ref_audio_path}")

            audio_output = await self._generate_audio_common(
                inference_method_name='inference_cross_lingual',
                text=text,
                # Parameters for CosyVoice2 `inference_cross_lingual`
                prompt_speech_16k=prompt_speech_16k, # Loaded audio tensor of the prompt
                stream=stream
            )

            if stream:
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else:
                output_mp3_path = TMP_DIR / f"tts_crosslingual_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(audio_output, output_mp3_path, SAMPLE_RATE)
                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                if output_mp3_path.exists():
                    try: os.unlink(output_mp3_path)
                    except OSError: logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except FileNotFoundError as e_fnf:
            logger.error(f"Cross-lingual TTS failed: Reference audio processing error - {e_fnf}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error with reference audio: {str(e_fnf)}")
        except Exception as e:
            logger.error(f"Cross-lingual TTS request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Cross-lingual TTS generation error: {str(e)}")


    async def tts_instruct(self, request: Request) -> Response:
        """
        Handles Instruction-based Text-to-Speech requests.
        Input: JSON with "text", "instruction", optionally "voice_type", "speed".
               Note: Streaming is typically NOT supported for instruct_tts by CosyVoice.
        Output: MP3 audio file.
        """
        try:
            payload = await request.json()
        except Exception:
            logger.warning("TTS Instruct: Invalid JSON payload.")
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")

        text = payload.get("text")
        instruction = payload.get("instruction") # Natural language instruction for style, emotion, etc.
        
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'text' field.")
        if not instruction or not isinstance(instruction, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'instruction' field.")

        voice_type = payload.get("voice_type", self.default_voice)
        # speed = float(payload.get("speed", 1.0)) # Speed not directly used

        if voice_type not in self.voice_prompts:
            logger.warning(f"TTS Instruct: Voice type '{voice_type}' not found. Using default '{self.default_voice}'.")
            voice_type = self.default_voice
            if voice_type not in self.voice_prompts:
                 raise HTTPException(status_code=400, detail=f"Default voice prompt '{self.default_voice}' is not available.")
        
        # CosyVoice's `inference_instruct` typically does not support streaming in the same way
        # as SFT or zero-shot, as the instruction might influence the entire utterance.
        # We will treat it as non-streaming here.
        try:
            # The `inference_instruct` method in CosyVoice2.
            # It expects text, instruction, and the speaker's name (instruct_dropdown).
            # Similar to SFT, the voice prompt audio is handled internally.
            audio_tensor = await self._generate_audio_common(
                inference_method_name='inference_instruct',
                text=text,
                instruct_text=instruction, # Parameter name in CosyVoice2
                instruct_dropdown=voice_type, # Parameter name in CosyVoice2
                stream=False # Instruct TTS usually generates the whole audio at once
            )

            if not isinstance(audio_tensor, torch.Tensor) or audio_tensor.numel() == 0:
                 raise Exception("Instruct TTS generation failed to produce audio tensor.")

            timestamp = int(time.time_ns() // 1000)
            output_mp3_path = TMP_DIR / f"tts_instruct_{timestamp}.mp3"
            await self.audio_processor.save_audio_to_mp3(audio_tensor, output_mp3_path, SAMPLE_RATE)
            
            with open(output_mp3_path, "rb") as f:
                audio_bytes = f.read()
            
            if output_mp3_path.exists():
                try: os.unlink(output_mp3_path)
                except OSError: logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
            
            return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Instruct TTS request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Instruct TTS generation error: {str(e)}")

    async def reconfigure(self, config: Dict) -> None:
        """
        Allows dynamic reconfiguration of the service if needed.
        For example, changing the default voice or updating model parameters.
        This is called by `serve.update_config()`.
        """
        logger.info(f"Reconfiguring service with new config: {config}")
        self.default_voice = config.get("default_voice", self.default_voice)
        # Add other reconfigurable parameters here
        # e.g., if model parameters can be changed without reloading the entire model.
        # If model needs reloading, that logic would be more complex.
        self.user_config.update(config)
        logger.info(f"Service reconfigured. New default voice: {self.default_voice}")


# --- Ray Serve Application Binding ---
# This binds the service class to a Ray Serve application, making it deployable.
# The application can then be run using `serve run api:cosyvoice_app`.
# We also add FastAPI/Starlette middleware here, like CORS.

# Define the Ray Serve application using Starlette for routing
# This allows defining routes like a FastAPI application.
# cosyvoice_app = CosyVoiceService.bind() # Basic binding

# For more complex routing and middleware, integrate with Starlette/FastAPI:
# Create a Starlette app and mount the Ray Serve deployment.
from starlette.applications import Starlette
from starlette.routing import Route

# Define routes for the Starlette application
# These routes will map to methods in the CosyVoiceService deployment.
# Ray Serve's `serve.ingress(app)` will automatically handle this mapping
# if the Starlette app is passed to the deployment's constructor.
# However, a more explicit way is to define a FastAPI/Starlette app
# and then bind the deployment to it.

# We will use the simpler approach of binding the class directly and then
# wrapping it with a FastAPI/Starlette app for middleware if needed,
# or by defining routes that call the deployment handle.

# Bind the deployment. This creates a handle that can be used to call the service.
cosyvoice_deployment = CosyVoiceService.options(
    # Pass any specific options for this binding if different from class level
).bind()


# Create a Starlette application to manage routes and middleware
# This application will be the main entry point for HTTP requests.
app = Starlette(
    routes=[
        Route("/tts", cosyvoice_deployment.tts_standard, methods=["POST"]),
        Route("/zero_shot_tts", cosyvoice_deployment.tts_zero_shot, methods=["POST"]),
        Route("/cross_lingual_tts", cosyvoice_deployment.tts_cross_lingual, methods=["POST"]),
        Route("/instruct_tts", cosyvoice_deployment.tts_instruct, methods=["POST"]),
        Route("/health", cosyvoice_deployment.health_check, methods=["GET"]),
        Route("/config", cosyvoice_deployment.get_config, methods=["GET"]),
    ],
    # Add exception handlers for cleaner error responses
    exception_handlers={
        HTTPException: lambda req, exc: JSONResponse({"detail": exc.detail}, status_code=exc.status_code),
        500: lambda req, exc: JSONResponse({"detail": "Internal server error"}, status_code=500),
    }
)

# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows web applications from different domains to access the API.
# Configure origins as needed for your environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins. For production, specify allowed domains.
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods.
    allow_headers=["*"], # Allows all headers.
)

# This 'cosyvoice_app' is what `serve run api:cosyvoice_app` will look for.
# It's the entry point for Ray Serve.
cosyvoice_app = app

# --- Main block for local testing (optional) ---
# This allows running the script directly with `python api.py` for local testing
# without `serve run`. This is useful for quick debugging of the service logic.
if __name__ == "__main__":
    logger.info("Starting Ray Serve locally for testing (api.py executed directly)...")
    
    # Initialize Ray if not already initialized (e.g., when running `python api.py`)
    if not ray.is_initialized():
        ray.init(
            # Configure local Ray cluster resources if needed
            # num_cpus=4, 
            # num_gpus=1 if torch.cuda.is_available() else 0,
            # logging_level=logging.WARNING, # Ray's internal logging
            # include_dashboard=False
        )

    # Deploy the application.
    # `serve.run` blocks until the script is killed.
    # The host and port here are for the Ray Serve HTTP proxy.
    serve.run(cosyvoice_app, host="0.0.0.0", port=8000, name="cosyvoice_serve_app")

    # To stop the service when running this way, you'd typically Ctrl+C.
    # Add cleanup if needed, though Ray's shutdown should handle actor cleanup.
    # try:
    #     while True:
    #         time.sleep(5)
    #         # print(serve.status()) # Periodically print status
    # except KeyboardInterrupt:
    #     logger.info("Shutting down Ray Serve...")
    # finally:
    #     serve.shutdown()
    #     if ray.is_initialized():
    #         ray.shutdown()
    #     logger.info("Ray Serve shut down complete.")

# Example of how to update config dynamically (e.g., from another script or admin endpoint)
# async def update_service_config(new_default_voice: str):
#    handle = serve.get_deployment("CosyVoiceAPI").get_handle()
#    await handle.reconfigure.remote({"default_voice": new_default_voice})
#    print(f"Sent reconfigure request. New default voice should be: {new_default_voice}")

# To run this application:
# 1. Ensure Ray is installed and a Ray cluster is running (or use `ray start --head`).
# 2. From the terminal in the project root: `serve run api:cosyvoice_app`
#    (or `python api.py` if __main__ block is used for local Ray start)
#
# To interact with the API (e.g., using curl or Postman):
# - Health check: GET http://localhost:8000/health
# - Config: GET http://localhost:8000/config
# - Standard TTS: POST http://localhost:8000/tts 
#   Body (JSON): {"text": "Hello world", "voice_type": "qwen"}
# - Zero-shot TTS: POST http://localhost:8000/zero_shot_tts
#   Body (form-data): text="Cloned voice.", reference_text="This is the reference.", reference_audio=@path/to/your/audio.wav
#
# Remember to manage dependencies (requirements.txt) and system packages (FFmpeg, Sox)
# as outlined in the Dockerfile and README.
