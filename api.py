"""
CosyVoice Ray Serve API

An HTTP API that serves the CosyVoice3 text-to-speech model
(``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``)
through Ray Serve. The whole service -- the ``CosyVoiceService`` deployment, the
``AudioProcessor`` helpers, the Starlette routing, and the ``cosyvoice_app`` entry
point -- lives in this single module.

Capabilities exposed over HTTP (all responses are MP3; pass ``"stream": true`` for a
chunked ``audio/mpeg`` response):

- ``POST /tts`` -- synthesize text with a built-in voice by routing through the model's
  prompt-based cross-lingual path using a bundled reference clip selected by
  ``voice_type``.
- ``POST /zero_shot_tts`` -- clone a voice from an uploaded reference clip plus its
  transcript (``inference_zero_shot``).
- ``POST /cross_lingual_tts`` -- clone a voice from an uploaded reference clip without
  a transcript, synthesizing text that may be in another language
  (``inference_cross_lingual``).
- ``POST /instruct_tts`` -- synthesize with a natural-language style instruction over a
  built-in voice (``inference_instruct2``).
- ``GET /health`` (aliased as ``GET /v1/model/cosyvoice/healthcheck``) and ``GET /config``.

CosyVoice3 produces 24 kHz audio (``SAMPLE_RATE``); reference clips are normalized
to 16 kHz mono (``REFERENCE_AUDIO_SAMPLE_RATE``) before inference. Deployment behavior --
autoscaling, GPU allocation, health-check and graceful-shutdown timeouts, default voice --
is driven entirely by environment variables read in the ``@serve.deployment`` decorator
and ``reconfigure()``; there is no config file.

The vendored ``cosyvoice/`` package is pinned to upstream CosyVoice commit
``074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`` and is treated as read-only; this module
adapts its ``AutoModel`` inference methods to HTTP.
"""

import logging
import os
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

# Configure CUDA environment for optimal performance and to avoid common issues.
# PCI_BUS_ID ensures CUDA device order matches nvidia-smi.
# CUDA_LAUNCH_BLOCKING can help with debugging CUDA errors by making them synchronous.
# PYTORCH_CUDA_ALLOC_CONF can fine-tune memory allocation.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to '1' for debugging CUDA errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import ray
import torch
import torchaudio
from modelscope import snapshot_download  # For downloading pre-trained models
from ray import serve
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware  # For enabling CORS
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Assuming CosyVoice modules are structured as per the original project
from cosyvoice.cli.cosyvoice import AutoModel

# Type hints for clarity
PathLike = str | Path
JsonDict = dict[str, Any]

# --- Configuration Constants ---
# Built-in voice prompts: maps a voice_type key to a reference clip in ASSET_DIR.
# These clips are used as the prompt audio for the /tts and /instruct_tts endpoints.
# Add a voice by dropping a .wav into asset/ and registering it here.
VOICE_PROMPTS: dict[str, str] = {"qwen": "qwen.wav"}
DEFAULT_VOICE: str = "qwen"  # Default voice key from VOICE_PROMPTS
SAMPLE_RATE: int = 24000  # CosyVoice3 synthesizes at 24 kHz (model output rate)
REFERENCE_AUDIO_SAMPLE_RATE: int = 16000  # Prompt/reference audio is normalized to 16 kHz mono
MODEL_FAMILY: str = "cosyvoice3"
MODEL_ID: str = os.environ.get("COSYVOICE_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
MODEL_DIR_NAME: str = os.environ.get("COSYVOICE_MODEL_DIR_NAME", "Fun-CosyVoice3-0.5B")
MODEL_REVISION: str = os.environ.get("COSYVOICE_MODEL_REVISION", "main")
END_OF_PROMPT: str = "<|endofprompt|>"
COSYVOICE3_SYSTEM_PROMPT: str = "You are a helpful assistant."

# --- Directory Setup ---
# Determine project root and other important directories
# __file__ refers to the current script (api.py)
ROOT_DIR: Path = Path(__file__).resolve().parent.absolute()
TMP_DIR: Path = ROOT_DIR / "tmp"
LOGS_DIR: Path = ROOT_DIR / "logs"
ASSET_DIR: Path = ROOT_DIR / "asset"
MODEL_DIR: Path = ROOT_DIR / "pretrained_models" / MODEL_DIR_NAME

# Create necessary directories if they don't exist
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR.parent, exist_ok=True)  # Ensure pretrained_models directory exists

# --- Logging Configuration ---
# Set up comprehensive logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Log to standard output
)

log_file_path: Path = LOGS_DIR / "cosyvoice_api.log"
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)  # Add file handler to root logger
logger = logging.getLogger(__name__)  # Get a logger for this specific module


def _format_cosyvoice3_prompt_text(text: str) -> str:
    """Return a CosyVoice3 text prompt with the required end-of-prompt marker."""
    text = text.strip()
    if END_OF_PROMPT in text:
        return text
    return f"{COSYVOICE3_SYSTEM_PROMPT}{END_OF_PROMPT}{text}"


def _format_cosyvoice3_instruction(instruction: str) -> str:
    """Return an instruction prompt compatible with CosyVoice3."""
    instruction = instruction.strip()
    if END_OF_PROMPT in instruction:
        return instruction
    return f"{COSYVOICE3_SYSTEM_PROMPT} {instruction}{END_OF_PROMPT}"


def _cosyvoice3_model_files_present(model_path: Path) -> bool:
    required_files = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt")
    return all((model_path / file_name).exists() for file_name in required_files)


# --- GPU and PyTorch Configuration ---
# Log PyTorch and CUDA details if CUDA is available
if torch.cuda.is_available():
    # It's good practice to initialize CUDA explicitly if you're managing devices.
    # However, PyTorch usually handles this. If issues arise, uncomment:
    # torch.cuda.init()
    torch.cuda.empty_cache()  # Clear any cached memory

    # Enable features for better performance on compatible hardware (e.g., Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Good for when input sizes don't vary much
    torch.backends.cudnn.deterministic = False  # Set to True if reproducibility is critical

    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"CUDA Device {i}: {props.name} (Total Memory: {props.total_memory / 1024**3:.1f}GB)"
        )

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
        target_sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE,
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

        logger.info(
            "Processing reference audio: "
            f"{input_audio_path} to {output_wav_path} at {target_sample_rate}Hz"
        )
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
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_audio_path),
                    "-ar",
                    str(target_sample_rate),
                    "-ac",
                    "1",
                    "-acodec",
                    "pcm_s16le",
                    str(output_wav_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ],
                check=True,  # Raise an exception for non-zero exit codes
                capture_output=True,  # Capture stdout and stderr
                text=True,  # Decode stdout/stderr as text
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
            return audio_tensor  # Or raise error

        # Ensure tensor is float32 for processing
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()

        # Handle potential multi-channel audio (e.g., stereo) by normalizing each channel
        if audio_tensor.dim() > 1:  # e.g., [channels, samples] or [batch, channels, samples]
            max_vals = audio_tensor.abs().max(dim=-1, keepdim=True)[0]
            # Avoid division by zero for silent channels
            audio_tensor = torch.where(max_vals > 1e-6, audio_tensor / max_vals, audio_tensor)
        else:  # Single channel audio [samples]
            max_val = audio_tensor.abs().max()
            if max_val > 1e-6:  # Avoid division by zero for silence
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

        if audio_tensor.numel() == 0:  # Check if tensor is empty
            logger.warning("Validation failed: Audio tensor is empty.")
            return False

        if torch.isnan(audio_tensor).any():  # Check for Not-a-Number values
            logger.warning("Validation failed: Audio tensor contains NaN values.")
            return False

        if torch.isinf(audio_tensor).any():  # Check for infinity values
            logger.warning("Validation failed: Audio tensor contains Inf values.")
            return False

        return True

    @staticmethod
    async def save_audio_to_mp3(
        audio_tensor: torch.Tensor, output_mp3_path: PathLike, sample_rate: int = SAMPLE_RATE
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
        timestamp = int(time.time_ns() // 1000)  # Microseconds for uniqueness
        temp_wav_path = TMP_DIR / f"temp_audio_{timestamp}.wav"

        try:
            # Ensure audio tensor is on CPU and in the correct format for torchaudio.save
            # Expected shape: [channels, samples] or [samples]
            if audio_tensor.is_cuda:
                audio_tensor = audio_tensor.cpu()
            if (
                audio_tensor.dim() == 1
            ):  # Add channel dimension if it's mono [samples] -> [1, samples]
                audio_tensor = audio_tensor.unsqueeze(0)

            torchaudio.save(
                str(temp_wav_path),
                audio_tensor,
                sample_rate,
                encoding="PCM_S",  # Signed 16-bit PCM
                bits_per_sample=16,
            )
            logger.info(f"Temporary WAV file saved: {temp_wav_path}")

            # Convert WAV to MP3 using FFmpeg
            # -i: Input file (temp WAV)
            # -codec:a libmp3lame: Use LAME MP3 encoder
            # -qscale:a 2: Variable bitrate encoding, quality level 2 (good quality)
            # -y: Overwrite output
            # -hide_banner -loglevel error: Suppress verbose output
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp_wav_path),
                    "-codec:a",
                    "libmp3lame",
                    "-qscale:a",
                    "2",  # VBR quality, 0-9 (lower is better)
                    str(output_mp3_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ],
                check=True,
                capture_output=True,
            )

            if not Path(output_mp3_path).exists() or Path(output_mp3_path).stat().st_size == 0:
                raise Exception(f"MP3 file failed to save or is empty: {output_mp3_path}")

            file_size_kb = Path(output_mp3_path).stat().st_size / 1024
            logger.info(f"Generated MP3 file: {output_mp3_path} (Size: {file_size_kb:.1f}KB)")

            # Basic sanity check for very small files (e.g., less than 1KB might indicate an issue)
            if file_size_kb < 0.5:
                logger.warning(
                    f"Generated MP3 file {output_mp3_path} is very small "
                    f"({file_size_kb:.1f}KB). This might indicate an issue."
                )

            return output_mp3_path
        except (RuntimeError, OSError) as e_torch:
            # torchaudio.save raises RuntimeError/OSError on encode or I/O failure.
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
    name="CosyVoiceAPI",  # Descriptive name for the deployment
    # Autoscaling configuration:
    # Adjust these based on expected load and available resources.
    autoscaling_config={
        "min_replicas": int(os.environ.get("MIN_REPLICAS", "1")),
        "initial_replicas": int(os.environ.get("INITIAL_REPLICAS", "1")),
        "max_replicas": int(
            os.environ.get("MAX_REPLICAS", "2")
        ),  # Max number of model server replicas
        "target_ongoing_requests": int(
            os.environ.get("TARGET_ONGOING_REQUESTS", "2")
        ),  # Target concurrent requests per replica
        "metrics_interval_s": 10.0,  # How often to scrape metrics
        "look_back_period_s": 30.0,  # Time window for averaging metrics
        "smoothing_factor": 1.0,  # Controls responsiveness of autoscaling (1.0 = no smoothing)
        # "downscale_delay_s": 600.0, # Delay before scaling down
        # "upscale_delay_s": 30.0,    # Delay before scaling up
    },
    # Resource allocation per replica:
    # Request GPUs if available and desired. Ray will manage allocation.
    # num_gpus=1 if torch.cuda.is_available() else 0
    ray_actor_options={"num_gpus": 1}
    if torch.cuda.is_available() and int(os.environ.get("NUM_GPUS_PER_REPLICA", "1")) > 0
    else {},
    # Health check configuration:
    health_check_period_s=int(
        os.environ.get("HEALTH_CHECK_PERIOD_S", "15")
    ),  # Frequency of health checks
    health_check_timeout_s=int(
        os.environ.get("HEALTH_CHECK_TIMEOUT_S", "30")
    ),  # Timeout for health check response
    # Request handling:
    max_ongoing_requests=int(
        os.environ.get("MAX_ONGOING_REQUESTS_PER_REPLICA", "5")
    ),  # Max concurrent requests a single replica handles
    max_queued_requests=int(
        os.environ.get("MAX_QUEUED_REQUESTS_DEPLOYMENT", "20")
    ),  # Max requests to queue at the deployment level
    # Graceful shutdown:
    graceful_shutdown_timeout_s=int(
        os.environ.get("GRACEFUL_SHUTDOWN_TIMEOUT_S", "60")
    ),  # Time for replicas to finish requests before shutdown
    graceful_shutdown_wait_loop_s=int(
        os.environ.get("GRACEFUL_SHUTDOWN_WAIT_LOOP_S", "5")
    ),  # Interval to check for request completion during shutdown
    # User-defined configuration accessible within the service
    user_config={
        "default_voice": DEFAULT_VOICE,
        "sample_rate": SAMPLE_RATE,
        "model_id": MODEL_ID,
        "model_dir_name": MODEL_DIR_NAME,
        "model_revision": MODEL_REVISION,
        "model_family": MODEL_FAMILY,
    },
)
class CosyVoiceService:
    """
    Ray Serve service class for CosyVoice TTS.
    Manages model loading, inference, and request handling.
    """

    def __init__(self, user_config: dict[str, Any] | None = None) -> None:
        """
        Initializes the CosyVoice service.
        This method is called once per replica when it's created.
        It handles device setup, model loading, and other one-time initializations.
        """
        self.user_config = user_config or {}
        self._reference_audio_cache: dict[
            str, PathLike
        ] = {}  # Cache for processed reference audio paths
        # Cache for built-in voice prompt WAV paths (keyed by voice_type).
        self._voice_prompt_paths: dict[str, PathLike] = {}

        self.device: str = self._setup_device()  # Determine 'cuda:X' or 'cpu'
        self.model: Any | None = None  # Placeholder for the TTS model
        self.audio_processor = AudioProcessor()  # Instantiate audio utility class

        # Load voice prompts from ASSET_DIR
        self.voice_prompts: dict[str, Path] = {
            key: ASSET_DIR / Path(filename).name  # Ensure only filename is used
            for key, filename in VOICE_PROMPTS.items()
        }
        self.default_voice: str = self.user_config.get("default_voice", DEFAULT_VOICE)

        # Ensure all specified voice prompt files exist
        for voice_key, prompt_path in self.voice_prompts.items():
            if not prompt_path.exists():
                logger.error(
                    f"Voice prompt file for '{voice_key}' not found at {prompt_path}. "
                    "This voice will be unavailable."
                )
                # Optionally, remove this voice from available prompts or raise an error
            else:
                logger.info(f"Voice prompt '{voice_key}' loaded from {prompt_path}")

        if not self.voice_prompts:
            logger.warning(
                f"No voice prompts found in {ASSET_DIR}. TTS with default prompts might fail."
            )
        elif self.default_voice not in self.voice_prompts:
            logger.warning(
                f"Default voice '{self.default_voice}' not found in available prompts. "
                "TTS may fail if no voice is specified."
            )

        self._download_and_setup_models()  # Download (if needed) and load the model

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
                gpu_ids = ray.get_gpu_ids()  # Returns a list of GPU IDs assigned to this actor
                if gpu_ids:
                    # Typically, one GPU is assigned per actor if num_gpus=1
                    assigned_gpu_id = int(gpu_ids[0])
                    device_str = f"cuda:{assigned_gpu_id}"

                    # Verify the device by trying to use it
                    # This also helps "warm up" the CUDA context for this GPU.
                    _ = torch.tensor([1.0]).to(device_str)

                    logger.info(
                        f"Replica assigned to GPU: {assigned_gpu_id}. Using device: {device_str}"
                    )
                    # Log memory details for this specific GPU
                    # torch.cuda.get_device_properties requires an int or torch.device
                    props = torch.cuda.get_device_properties(assigned_gpu_id)
                    logger.info(
                        f"Properties for {device_str}: {props.name}, "
                        f"Memory: {props.total_memory / 1024**3:.1f}GB"
                    )
                    logger.info(
                        f"Initial memory allocated on {device_str}: "
                        f"{torch.cuda.memory_allocated(device_str) / 1024**2:.1f}MB"
                    )
                    return device_str
                else:
                    logger.warning(
                        "CUDA is available, but no GPUs were assigned to this replica by Ray. "
                        "Falling back to CPU."
                    )
                    return "cpu"
            except Exception as e:
                logger.error(
                    f"Error detecting or setting up assigned GPU: {e}. Falling back to CPU."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Attempt to clear cache if error occurred
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
            if hasattr(self, "model") and self.model is not None:
                # If the model has a specific cleanup method, call it.
                # e.g., if self.model.cleanup(): self.model.cleanup()
                del self.model  # Remove reference to allow garbage collection
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
        Downloads the CosyVoice pre-trained model from ModelScope if it does not exist locally,
        then loads the model onto the configured device.
        """
        local_model_path = MODEL_DIR

        try:
            if not _cosyvoice3_model_files_present(local_model_path):
                logger.info(
                    f"CosyVoice3 model files not found at {local_model_path}. "
                    f"Downloading from ModelScope ({MODEL_ID}, revision={MODEL_REVISION})..."
                )
                snapshot_download(
                    MODEL_ID,
                    local_dir=str(local_model_path),
                    revision=MODEL_REVISION,
                )
                logger.info(f"Model download completed to {local_model_path}.")
            else:
                logger.info(f"CosyVoice3 model found at {local_model_path}. Skipping download.")

            logger.info(
                f"Initializing CosyVoice3 model on device: {self.device} "
                f"from path: {local_model_path}"
            )

            use_cuda_features = self.device.startswith("cuda")

            self.model = AutoModel(
                model_dir=str(local_model_path),
                load_trt=False,
                load_vllm=False,
                fp16=use_cuda_features,
            )
            logger.info(f"Successfully initialized CosyVoice3 model on {self.device}.")

            if use_cuda_features and torch.cuda.is_available():
                # Log memory usage after model loading
                allocated_mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                reserved_mem_mb = torch.cuda.memory_reserved(self.device) / 1024**2
                logger.info(
                    f"GPU memory after model init on {self.device}: "
                    f"Allocated={allocated_mem_mb:.1f}MB, "
                    f"Reserved={reserved_mem_mb:.1f}MB"
                )

        except Exception as e:
            logger.error(f"Failed to download or initialize CosyVoice model: {e}", exc_info=True)
            # If GPU initialization failed, attempt to fall back to CPU if not already on CPU.
            if self.device.startswith("cuda"):
                logger.warning(
                    "Model initialization on GPU failed. Attempting to fall back to CPU."
                )
                self.device = "cpu"  # Switch device to CPU
                try:
                    self.model = AutoModel(
                        model_dir=str(local_model_path),
                        load_trt=False,
                        load_vllm=False,
                        fp16=False,  # FP16 usually not for CPU
                    )
                    logger.info(
                        "Successfully initialized CosyVoice3 model on CPU after GPU fallback."
                    )
                except Exception as e_cpu:
                    logger.error(
                        f"Failed to initialize CosyVoice model on CPU after fallback: {e_cpu}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        "CosyVoice model could not be loaded on GPU or CPU."
                    ) from e_cpu
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
        except OSError:  # Handle potential race condition if file is deleted
            mod_time = time.time()
        cache_key = f"{str(input_path)}_{mod_time}"

        if cache_key in self._reference_audio_cache:
            cached_path = self._reference_audio_cache[cache_key]
            if Path(cached_path).exists():  # Ensure cached file still exists
                logger.info(f"Using cached processed reference audio: {cached_path}")
                return cached_path
            else:
                logger.warning(f"Cached reference audio {cached_path} not found. Re-processing.")
                del self._reference_audio_cache[cache_key]  # Remove stale entry

        # If not in cache or stale, process the audio
        # Create a unique name for the processed file in TMP_DIR
        timestamp = int(time.time_ns() // 1000)
        processed_filename = f"processed_ref_{input_path.stem}_{timestamp}.wav"
        processed_output_path = TMP_DIR / processed_filename

        logger.info(f"Normalizing reference audio {input_path} to 16 kHz mono WAV...")
        try:
            self.audio_processor.process_reference_audio(
                input_audio_path=input_path,
                output_wav_path=processed_output_path,
                target_sample_rate=REFERENCE_AUDIO_SAMPLE_RATE,
            )
            self._reference_audio_cache[cache_key] = processed_output_path  # Add to cache
            logger.info(f"Reference audio processed and cached: {processed_output_path}")
            return processed_output_path
        except Exception as e:
            logger.error(f"Failed to process reference audio {input_path}: {e}", exc_info=True)
            # Ensure partially created file is removed on error
            if processed_output_path.exists():
                try:
                    os.unlink(processed_output_path)
                except OSError:
                    pass
            raise  # Re-raise the exception to be handled by the caller

    async def _get_voice_prompt_path(self, voice_type: str) -> PathLike:
        """
        Returns a built-in voice prompt clip as a normalized 16 kHz mono WAV path.

        The /tts and /instruct_tts endpoints supply a bundled reference clip selected by
        ``voice_type`` as the prompt for the model's prompt-based inference. The clip is
        normalized to REFERENCE_AUDIO_SAMPLE_RATE via FFmpeg and cached.

        Args:
            voice_type: A key from ``self.voice_prompts`` (e.g., the default ``"qwen"``).

        Returns:
            The processed prompt audio path.
        """
        cached = self._voice_prompt_paths.get(voice_type)
        if cached is not None:
            return cached

        prompt_path = self.voice_prompts[voice_type]
        processed_path = await self._get_processed_reference_audio(prompt_path)
        self._voice_prompt_paths[voice_type] = processed_path
        logger.info(f"Loaded built-in voice prompt '{voice_type}' from {prompt_path}.")
        return processed_path

    async def _generate_audio_common(
        self, inference_method_name: str, text: str, stream: bool, **kwargs
    ) -> torch.Tensor | AsyncGenerator[bytes, None]:
        """
        Common logic for generating audio using different CosyVoice inference methods.
        Handles streaming and non-streaming output.

        Args:
            inference_method_name: Name of the CosyVoice inference method to call (e.g.,
                                   'inference_zero_shot', 'inference_cross_lingual',
                                   'inference_instruct2').
            text: The input text to synthesize (passed positionally as ``tts_text``).
            stream: Whether the HTTP response streams. Non-streaming requests drive the model
                    with ``stream=False`` so model-level speed control remains active.
            **kwargs: Method-specific arguments forwarded to the model (e.g., ``prompt_text``,
                      ``prompt_wav``, ``instruct_text``, ``speed``).

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
            raise AttributeError(
                f"Inference method '{inference_method_name}' not found in CosyVoice model."
            )

        logger.info(
            f"Starting TTS generation with method '{inference_method_name}' "
            f'for text: "{text[:50]}..." (Stream: {stream})'
        )

        # The 'stream' argument for CosyVoice model methods controls audio chunking.
        inference_args = {**kwargs, "stream": stream}

        # --- Streaming Logic ---
        if stream:

            async def stream_generator():
                temp_files_to_clean = []
                try:
                    chunk_index = 0
                    for result_chunk in inference_fn(text, **inference_args):
                        if "tts_speech" in result_chunk:
                            audio_tensor_chunk = result_chunk["tts_speech"]

                            if not self.audio_processor.validate_audio_tensor(audio_tensor_chunk):
                                logger.warning(
                                    f"Stream chunk {chunk_index}: Invalid audio tensor received, "
                                    "skipping."
                                )
                                continue

                            # Normalize audio chunk
                            audio_tensor_chunk = self.audio_processor.normalize_audio_tensor(
                                audio_tensor_chunk
                            )

                            # Save chunk to a temporary MP3 file
                            timestamp = int(time.time_ns() // 1000)
                            temp_mp3_chunk_path = (
                                TMP_DIR / f"stream_chunk_{timestamp}_{chunk_index}.mp3"
                            )
                            temp_files_to_clean.append(temp_mp3_chunk_path)

                            await self.audio_processor.save_audio_to_mp3(
                                audio_tensor_chunk, temp_mp3_chunk_path, sample_rate=SAMPLE_RATE
                            )

                            # Yield the content of the MP3 chunk file
                            with open(temp_mp3_chunk_path, "rb") as f_chunk:
                                yield f_chunk.read()

                            logger.debug(
                                f"Streamed MP3 chunk {chunk_index} "
                                f"({temp_mp3_chunk_path.stat().st_size} bytes)"
                            )
                            chunk_index += 1

                    if chunk_index == 0:
                        logger.warning(
                            f"TTS generation for method '{inference_method_name}' produced "
                            "no audio chunks."
                        )
                        # Optionally, yield a small silent MP3 or raise an error for the client
                        # For now, it will just end the stream.
                except Exception as e_stream:
                    logger.error(
                        "Error during streaming TTS generation "
                        f"({inference_method_name}): {e_stream}",
                        exc_info=True,
                    )
                    # If an error occurs, the client connection might be broken.
                    # Signaling this after response streaming begins is protocol-dependent.
                finally:
                    # Clean up all temporary chunk files
                    for temp_file in temp_files_to_clean:
                        if temp_file.exists():
                            try:
                                os.unlink(temp_file)
                            except OSError:
                                logger.warning(f"Failed to delete temp stream chunk: {temp_file}")
                    logger.info(
                        f"Finished streaming for '{inference_method_name}'. "
                        f"Cleaned {len(temp_files_to_clean)} temp files."
                    )

            return stream_generator()  # Return the async generator for streaming response

        # --- Non-Streaming Logic ---
        else:
            collected_audio_tensors: list[torch.Tensor] = []
            chunk_index = 0
            for result_chunk in inference_fn(text, **inference_args):
                if "tts_speech" in result_chunk:
                    audio_tensor_chunk = result_chunk["tts_speech"]
                    if not self.audio_processor.validate_audio_tensor(audio_tensor_chunk):
                        logger.warning(
                            f"Non-stream chunk {chunk_index}: Invalid audio tensor, skipping."
                        )
                        continue

                    audio_tensor_chunk = self.audio_processor.normalize_audio_tensor(
                        audio_tensor_chunk
                    )
                    collected_audio_tensors.append(
                        audio_tensor_chunk.cpu()
                    )  # Move to CPU before collecting
                    chunk_index += 1

            if not collected_audio_tensors:
                logger.error(
                    f"TTS generation for method '{inference_method_name}' produced "
                    "no valid audio output."
                )
                raise Exception("TTS generation failed to produce audio.")

            # Concatenate all audio chunks into a single tensor
            # Ensure they are all on the same device (CPU) and have compatible shapes
            # CosyVoice usually outputs [1, num_samples] or [num_samples]
            # If shapes are [1, N], [1, M], etc., cat along dim 1.
            # If shapes are [N], [M], etc., cat along dim 0.
            # Chunks are [num_samples] or [1, num_samples]; concatenate on sample dimension.
            # CosyVoice outputs tensors that can be concatenated along the last dimension.
            try:
                full_audio_tensor = torch.cat(collected_audio_tensors, dim=-1)
                logger.info(
                    f"Concatenated {len(collected_audio_tensors)} audio chunks for "
                    f"'{inference_method_name}'. Final duration: "
                    f"{full_audio_tensor.shape[-1] / SAMPLE_RATE:.2f}s"
                )
                return full_audio_tensor
            except Exception as e_cat:
                logger.error(
                    f"Error concatenating audio chunks for '{inference_method_name}': {e_cat}",
                    exc_info=True,
                )
                # Log shapes for debugging
                for i, t in enumerate(collected_audio_tensors):
                    logger.error(
                        f"Chunk {i} shape: {t.shape}, dtype: {t.dtype}, device: {t.device}"
                    )
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
            return JSONResponse(
                {"status": "unhealthy", "reason": "Model not loaded"}, status_code=503
            )

        # Optionally, perform a quick inference test if feasible and not too resource-intensive.
        # For now, just checking model presence is sufficient.
        return JSONResponse(
            {
                "status": "healthy",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "device": self.device,
                "model_loaded": self.model is not None,
                "ray_actor_id": ray.get_runtime_context().actor_id.hex()
                if ray.is_initialized()
                else "N/A",
                "ray_node_id": ray.get_runtime_context().node_id.hex()
                if ray.is_initialized()
                else "N/A",
            }
        )

    async def get_config(self, request: Request) -> JSONResponse:
        """
        Returns the current configuration of the service.
        """
        logger.info("Configuration request received.")
        return JSONResponse(
            {
                "default_voice": self.default_voice,
                "available_voice_prompts": list(self.voice_prompts.keys()),
                "sample_rate": SAMPLE_RATE,
                "reference_audio_sample_rate": REFERENCE_AUDIO_SAMPLE_RATE,
                "device_in_use": self.device,
                "model_directory": str(MODEL_DIR),  # Expose where models are expected
                "model_id": MODEL_ID,
                "model_dir_name": MODEL_DIR_NAME,
                "model_revision": MODEL_REVISION,
                "model_family": MODEL_FAMILY,
                "max_replicas_config": self.user_config.get(
                    "max_replicas", os.environ.get("MAX_REPLICAS", "2")
                ),  # Example of exposing deployment config
            }
        )

    async def tts_standard(self, request: Request) -> Response:
        """
        Handles standard Text-to-Speech requests.
        Input: JSON with "text", optionally "voice_type", "speed", "stream".
        Output: MP3 audio file or streaming audio.
        """
        try:
            payload = await request.json()
        except Exception as exc:
            logger.warning("TTS Standard: Invalid JSON payload received.")
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

        text = payload.get("text")
        if not text or not isinstance(text, str):
            raise HTTPException(
                status_code=400, detail="Missing or invalid 'text' field in payload."
            )

        voice_type = payload.get("voice_type", self.default_voice)
        stream = bool(payload.get("stream", False))
        try:
            speed = float(payload.get("speed", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="'speed' must be a number.") from None

        if voice_type not in self.voice_prompts:
            logger.warning(
                f"TTS Standard: Voice type '{voice_type}' not found. "
                f"Using default '{self.default_voice}'."
            )
            voice_type = self.default_voice  # Fallback to default if specified voice is invalid
            if voice_type not in self.voice_prompts:  # If default is also missing
                raise HTTPException(
                    status_code=400,
                    detail=f"Default voice prompt '{self.default_voice}' is not available.",
                )

        try:
            prompt_wav = await self._get_voice_prompt_path(voice_type)
            audio_output = await self._generate_audio_common(
                inference_method_name="inference_cross_lingual",
                text=_format_cosyvoice3_prompt_text(text),
                prompt_wav=str(prompt_wav),  # Built-in voice as the cloning prompt
                speed=speed,
                stream=stream,
            )

            if stream:  # Output is an AsyncGenerator
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else:  # Output is a torch.Tensor
                timestamp = int(time.time_ns() // 1000)
                output_mp3_path = TMP_DIR / f"tts_standard_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(
                    audio_output, output_mp3_path, SAMPLE_RATE
                )

                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()

                # Clean up the generated MP3 file
                if output_mp3_path.exists():
                    try:
                        os.unlink(output_mp3_path)
                    except OSError:
                        logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")

                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:  # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            logger.error(f"TTS Standard request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"TTS generation error: {str(e)}") from e

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
            reference_text = form_data.get("reference_text")  # Transcript of the reference audio
            reference_audio_file: UploadFile | None = form_data.get("reference_audio")
            stream = str(form_data.get("stream", "false")).lower() == "true"
            try:
                speed = float(form_data.get("speed", 1.0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="'speed' must be a number.") from None

            if not all([text, reference_text, reference_audio_file]):
                missing_fields = [
                    f
                    for f, v in {
                        "text": text,
                        "reference_text": reference_text,
                        "reference_audio": reference_audio_file,
                    }.items()
                    if not v
                ]
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required form fields: {', '.join(missing_fields)}",
                )

            if not isinstance(text, str) or not isinstance(reference_text, str):
                raise HTTPException(
                    status_code=400, detail="Fields 'text' and 'reference_text' must be strings."
                )

            # Save uploaded reference audio to a temporary file
            timestamp = int(time.time_ns() // 1000)
            upload_name = Path(reference_audio_file.filename or "reference_audio").name
            temp_ref_audio_path = TMP_DIR / f"ref_audio_upload_{timestamp}_{upload_name}"

            try:
                with open(temp_ref_audio_path, "wb") as f_ref:
                    contents = await reference_audio_file.read()
                    f_ref.write(contents)
                logger.info(
                    f"Zero-shot: Reference audio saved to temporary file: {temp_ref_audio_path}"
                )

                # Convert the reference audio to the normalized WAV path used by CosyVoice.
                processed_ref_audio_path = await self._get_processed_reference_audio(
                    temp_ref_audio_path
                )

            finally:
                # Clean up the initially uploaded temporary reference audio file
                if temp_ref_audio_path.exists():
                    try:
                        os.unlink(temp_ref_audio_path)
                    except OSError:
                        logger.warning(
                            f"Failed to delete uploaded temp ref audio: {temp_ref_audio_path}"
                        )
                # Note: The *processed* reference audio file from `_get_processed_reference_audio`
                # is managed by the cache and its cleanup logic.

            audio_output = await self._generate_audio_common(
                inference_method_name="inference_zero_shot",
                text=text,
                prompt_text=_format_cosyvoice3_prompt_text(reference_text),
                prompt_wav=str(processed_ref_audio_path),
                speed=speed,
                stream=stream,
            )

            if stream:
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else:
                output_mp3_path = TMP_DIR / f"tts_zeroshot_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(
                    audio_output, output_mp3_path, SAMPLE_RATE
                )
                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                if output_mp3_path.exists():
                    try:
                        os.unlink(output_mp3_path)
                    except OSError:
                        logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except (
            FileNotFoundError
        ) as e_fnf:  # Specifically for reference audio not found by _get_processed_reference_audio
            logger.error(
                f"Zero-shot TTS failed: Reference audio processing error - {e_fnf}", exc_info=True
            )
            raise HTTPException(
                status_code=400, detail=f"Error with reference audio: {str(e_fnf)}"
            ) from e_fnf
        except Exception as e:
            logger.error(f"Zero-shot TTS request failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Zero-shot TTS generation error: {str(e)}"
            ) from e

    async def tts_cross_lingual(self, request: Request) -> Response:
        """
        Handles Cross-Lingual Text-to-Speech requests.
        Input: FormData with "text", "reference_audio" (file), optionally "speed", "stream".
        Output: MP3 audio file or streaming audio.
        """
        try:
            form_data = await request.form()
            text = form_data.get("text")
            reference_audio_file: UploadFile | None = form_data.get("reference_audio")
            stream = str(form_data.get("stream", "false")).lower() == "true"
            try:
                speed = float(form_data.get("speed", 1.0))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="'speed' must be a number.") from None

            if not text or not reference_audio_file:
                missing_fields = [
                    f
                    for f, v in {"text": text, "reference_audio": reference_audio_file}.items()
                    if not v
                ]
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required form fields: {', '.join(missing_fields)}",
                )

            if not isinstance(text, str):
                raise HTTPException(status_code=400, detail="Field 'text' must be a string.")

            timestamp = int(time.time_ns() // 1000)
            upload_name = Path(reference_audio_file.filename or "reference_audio").name
            temp_ref_audio_path = TMP_DIR / f"ref_audio_crosslingual_{timestamp}_{upload_name}"

            try:
                with open(temp_ref_audio_path, "wb") as f_ref:
                    contents = await reference_audio_file.read()
                    f_ref.write(contents)
                logger.info(f"Cross-lingual: Reference audio saved to: {temp_ref_audio_path}")

                processed_ref_audio_path = await self._get_processed_reference_audio(
                    temp_ref_audio_path
                )
            finally:
                if temp_ref_audio_path.exists():
                    try:
                        os.unlink(temp_ref_audio_path)
                    except OSError:
                        logger.warning(
                            f"Failed to delete uploaded temp ref audio: {temp_ref_audio_path}"
                        )

            audio_output = await self._generate_audio_common(
                inference_method_name="inference_cross_lingual",
                text=_format_cosyvoice3_prompt_text(text),
                prompt_wav=str(processed_ref_audio_path),
                speed=speed,
                stream=stream,
            )

            if stream:
                return StreamingResponse(audio_output, media_type="audio/mpeg")
            else:
                output_mp3_path = TMP_DIR / f"tts_crosslingual_{timestamp}.mp3"
                await self.audio_processor.save_audio_to_mp3(
                    audio_output, output_mp3_path, SAMPLE_RATE
                )
                with open(output_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                if output_mp3_path.exists():
                    try:
                        os.unlink(output_mp3_path)
                    except OSError:
                        logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")
                return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except FileNotFoundError as e_fnf:
            logger.error(
                f"Cross-lingual TTS failed: Reference audio processing error - {e_fnf}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400, detail=f"Error with reference audio: {str(e_fnf)}"
            ) from e_fnf
        except Exception as e:
            logger.error(f"Cross-lingual TTS request failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Cross-lingual TTS generation error: {str(e)}"
            ) from e

    async def tts_instruct(self, request: Request) -> Response:
        """
        Handles instruction-based Text-to-Speech requests.

        CosyVoice3 steers style/emotion via ``inference_instruct2``, which needs prompt audio,
        so the built-in voice (``voice_type``) supplies the timbre and the instruction shapes
        delivery. Plain client instructions are wrapped with the CosyVoice3 end-of-prompt
        marker for backward compatibility.

        Input: JSON with "text", "instruction", optionally "voice_type", "speed", "stream".
        Output: MP3 audio file, or a chunked audio/mpeg stream when "stream" is true.
        """
        try:
            payload = await request.json()
        except Exception as exc:
            logger.warning("TTS Instruct: Invalid JSON payload.")
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

        text = payload.get("text")
        instruction = payload.get(
            "instruction"
        )  # Natural language instruction for style, emotion, etc.

        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'text' field.")
        if not instruction or not isinstance(instruction, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'instruction' field.")

        voice_type = payload.get("voice_type", self.default_voice)
        stream = bool(payload.get("stream", False))
        try:
            speed = float(payload.get("speed", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="'speed' must be a number.") from None

        if voice_type not in self.voice_prompts:
            logger.warning(
                f"TTS Instruct: Voice type '{voice_type}' not found. "
                f"Using default '{self.default_voice}'."
            )
            voice_type = self.default_voice
            if voice_type not in self.voice_prompts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Default voice prompt '{self.default_voice}' is not available.",
                )

        try:
            prompt_wav = await self._get_voice_prompt_path(voice_type)
            audio_output = await self._generate_audio_common(
                inference_method_name="inference_instruct2",
                text=text,
                instruct_text=_format_cosyvoice3_instruction(instruction),
                prompt_wav=str(prompt_wav),  # Built-in voice as the cloning prompt
                speed=speed,
                stream=stream,
            )

            if stream:
                return StreamingResponse(audio_output, media_type="audio/mpeg")

            timestamp = int(time.time_ns() // 1000)
            output_mp3_path = TMP_DIR / f"tts_instruct_{timestamp}.mp3"
            await self.audio_processor.save_audio_to_mp3(audio_output, output_mp3_path, SAMPLE_RATE)

            with open(output_mp3_path, "rb") as f:
                audio_bytes = f.read()

            if output_mp3_path.exists():
                try:
                    os.unlink(output_mp3_path)
                except OSError:
                    logger.warning(f"Failed to delete temp MP3: {output_mp3_path}")

            return Response(audio_bytes, media_type="audio/mpeg")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Instruct TTS request failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Instruct TTS generation error: {str(e)}"
            ) from e

    async def reconfigure(self, config: dict) -> None:
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
# Bind the deployment to obtain a handle, then expose it over HTTP through a Starlette app
# whose routes call the deployment's methods. `serve run api:cosyvoice_app` and the __main__
# block below both resolve the module-level `cosyvoice_app` defined further down.

# Bind the deployment. This creates a handle used to dispatch requests to replicas.
cosyvoice_deployment = CosyVoiceService.options(
    # Pass any specific options for this binding if different from the class-level decorator.
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
        Route(
            "/v1/model/cosyvoice/healthcheck", cosyvoice_deployment.health_check, methods=["GET"]
        ),
        Route("/config", cosyvoice_deployment.get_config, methods=["GET"]),
    ],
    # Add exception handlers for cleaner error responses
    exception_handlers={
        HTTPException: lambda req, exc: JSONResponse(
            {"detail": exc.detail}, status_code=exc.status_code
        ),
        500: lambda req, exc: JSONResponse({"detail": "Internal server error"}, status_code=500),
    },
)

# Add CORS (Cross-Origin Resource Sharing) middleware
# This allows web applications from different domains to access the API.
# Configure origins as needed for your environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, specify allowed domains.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
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
    # SERVE_PORT lets Docker/deployments override the bind port (defaults to 8000 for local runs).
    serve_port = int(os.environ.get("SERVE_PORT", "8000"))
    serve.run(cosyvoice_app, host="0.0.0.0", port=serve_port, name="cosyvoice_serve_app")

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
# 1. Ensure a Ray cluster is available (or use `ray start --head`).
# 2. From the project root: `serve run api:cosyvoice_app`
#    (or `python api.py`, which starts a local single-node Ray cluster).
# The HTTP proxy binds to SERVE_PORT (default 8000 locally; the Docker image sets 9998).
#
# To interact with the API (curl/Postman): all responses are MP3; add "stream": true for a
# chunked audio/mpeg stream, and an optional "speed" (float, 1.0 = normal):
# - Health:        GET  http://localhost:8000/health
# - Config:        GET  http://localhost:8000/config
# - Standard TTS:  POST http://localhost:8000/tts
#     JSON: {"text": "Hello world", "voice_type": "qwen"}
# - Zero-shot:     POST http://localhost:8000/zero_shot_tts
#     form-data: text=..., reference_text=<transcript>, reference_audio=@sample.wav
# - Cross-lingual: POST http://localhost:8000/cross_lingual_tts
#     form-data: text=<target-language text>, reference_audio=@sample.wav
# - Instruct:      POST http://localhost:8000/instruct_tts
#     JSON: {"text": "...", "instruction": "speak cheerfully", "voice_type": "qwen"}
#
# Dependencies live in requirements.txt; system packages (FFmpeg, Sox) and the runtime
# are set up by the Dockerfile (see README for details).
