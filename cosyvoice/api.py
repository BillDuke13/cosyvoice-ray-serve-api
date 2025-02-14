# Configure CUDA environment before importing PyTorch
import os
import sys
import json
import time
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import ray
from ray import serve
import torch
import torchaudio

# Initialize CUDA before importing torch
import torch.backends.cuda
import torch.backends.cudnn

# Dynamically configure CUDA devices
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(i) for i in range(gpu_count)
    )
else:
    # Disable CUDA if no GPUs found
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Make device IDs match nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# More detailed error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Optimize memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Enable CUDA and cuDNN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Check CUDA environment
if torch.cuda.is_available():
    # Force PyTorch to reinitialize CUDA state
    torch.cuda.init()
    torch.cuda.empty_cache()

    # Log CUDA information
    logger = logging.getLogger(__name__)
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"CUDA Device {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)"
        )

    # Set default CUDA device
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        logger.info(f"Default CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    logger = logging.getLogger(__name__)
    logger.warning("CUDA is not available. Running on CPU only.")

from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

"""CosyVoice Text-to-Speech Service API.

This module provides a Ray Serve based HTTP API for text-to-speech synthesis
using CosyVoice models. It supports standard TTS, voice cloning, and
cross-lingual synthesis with high-performance audio generation capabilities.
"""

from __future__ import annotations

# Standard library imports
import atexit
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import ray
from ray import serve
import torch
import torchaudio
from modelscope import snapshot_download
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

# Local imports
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Type aliases
PathLike = Union[str, Path]
JsonDict = Dict[str, Any]

# Constants
# Mapping of voice types to their prompt audio files.
VOICE_PROMPTS = {
    "qwen": "asset/qwen.wav",  # Default qwen voice
}
DEFAULT_VOICE = "qwen"


class AudioProcessor:
    """Utility class for processing and converting audio files.

    This class provides static methods for handling audio file operations such
    as format conversion and sample rate adjustment. It uses FFmpeg for audio
    processing to ensure high-quality conversions and standardized output
    formats.
    """

    @staticmethod
    def process_reference_audio(
        audio_path: PathLike,
        output_path: PathLike,
        sample_rate: int = 16000
    ) -> None:
        """Processes a reference audio file to match the required format.

        Converts the input audio file to WAV format with the specified sample
        rate using FFmpeg. The output file will be single-channel (mono) and
        use 16-bit PCM encoding.

        Args:
            audio_path: Path to the source audio file. Can be any format
              supported by FFmpeg.
            output_path: Path where the processed audio will be saved (WAV
              format).
            sample_rate: Target sample rate in Hz. Defaults to 16000.

        Raises:
            subprocess.CalledProcessError: If FFmpeg processing fails.
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Input audio file not found: {audio_path}"
            )

        try:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-ignore_unknown", "-y",
                    "-i", str(audio_path),
                    "-ar", str(sample_rate),
                    "-ac", "1",  # Convert to mono
                    "-acodec", "pcm_s16le",  # 16-bit PCM
                    str(output_path)
                ],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg processing failed: {e.stderr}")
            raise

# Set up paths
root_dir = Path(__file__).resolve().parent.absolute()
project_dir = root_dir.parent  # Go up one level to project root
tmp_dir = project_dir / 'tmp'
logs_dir = project_dir / 'logs'
asset_dir = project_dir / 'asset'  # Add asset directory
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(asset_dir, exist_ok=True)

# Convert to string paths
root_dir = str(root_dir)
tmp_dir = str(tmp_dir)
logs_dir = str(logs_dir)
asset_dir = str(asset_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{logs_dir}/{datetime.datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure environment for Unix systems
os.environ['PATH'] = f"{root_dir}:{root_dir}/ffmpeg:" + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')

# Initialize CUDA if available
if torch.cuda.is_available():
    try:
        # Test CUDA availability
        test_tensor = torch.cuda.FloatTensor(1)
        del test_tensor
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA-capable GPU(s)")
        for i in range(gpu_count):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**3:.1f}GB")
    except Exception as e:
        logger.warning(f"Failed to initialize CUDA: {e}")
        logger.info("Will run on CPU")

@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1 if torch.cuda.is_available() else 0,  # Request one GPU per replica
    },
    max_ongoing_requests=100,  # Control concurrency
    health_check_period_s=10,  # Regular health checks
    health_check_timeout_s=30,   # Health check timeout
    graceful_shutdown_timeout_s=60 # Allow time for graceful shutdown
)
class CosyVoiceService:
    """Ray Serve deployment for CosyVoice text-to-speech service.

    This class implements a Ray Serve deployment that provides
    text-to-speech synthesis services using CosyVoice models. It supports
    multiple synthesis modes including:

    - Standard text-to-speech synthesis
    - Cross-lingual voice cloning
    - Same-language voice cloning

    The service automatically detects and utilizes available GPU resources for
    optimal performance, with graceful fallback to CPU when needed. It includes
    built-in resource management, automatic cleanup, and health monitoring
    capabilities.
    """

    def __init__(self) -> None:
        """Initialize models and resources.
        
        Attributes:
            gpu_id (Optional[int]): ID of the GPU device assigned to this service instance.
            device (str): Device to run models on ('cuda:<id>' or 'cpu').
            sft_model (CosyVoice): Fine-tuned CosyVoice model for standard TTS.
            tts_model (CosyVoice2): CosyVoice2 model for advanced TTS operations.
            audio_processor (AudioProcessor): Utility for audio file processing.
            voice_prompts (Dict[str, str]): Available voice types and their prompt files.
            default_voice (str): Default voice type to use.
        """
        if torch.cuda.is_available():
            self.gpu_id = ray.get_gpu_ids()[0]
            torch.cuda.set_device(self.gpu_id)
            self.device = f"cuda:{self.gpu_id}"
        else:
            self.device = "cpu"
        self._setup_models()
        self.audio_processor = AudioProcessor()
        self.voice_prompts = VOICE_PROMPTS
        self.default_voice = DEFAULT_VOICE

        # Register cleanup handler
        ray.worker._post_init_hooks.append(self._cleanup_resources)

    def _cleanup_resources(self) -> None:
        """Clean up resources when the deployment is stopped."""
        try:
            # Clear GPU memory if using CUDA
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                try:
                    # Move models to CPU before clearing GPU memory
                    if hasattr(self, 'sft_model'):
                        self.sft_model = self.sft_model.cpu()
                    if hasattr(self, 'tts_model'):
                        self.tts_model = self.tts_model.cpu()
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared GPU {self.gpu_id} memory")
                except Exception as e:
                    logger.error(f"Error clearing GPU memory: {e}")

            # Delete model references
            if hasattr(self, 'sft_model'):
                del self.sft_model
            if hasattr(self, 'tts_model'):
                del self.tts_model

            logger.info("Cleaned up deployment resources")
        except Exception as e:
            logger.error(f"Error during deployment cleanup: {e}")

    async def __call__(self, request: Request) -> Response:
        """Handle incoming HTTP requests.

        Args:
            request: The incoming HTTP request.

        Returns:
            HTTP response containing audio data or error message.
        """
        try:
            if request.method == "POST":
                if request.headers.get("content-type") == "application/json":
                    params = await request.json()
                else:
                    form = await request.form()
                    params = dict(form)

                # Validate and transform parameters
                params = self._validate_params(params)

                # Extract the endpoint from the path
                path = request.url.path.strip('/')
                # Default to tts if no path
                endpoint = path.split('/')[-1] if path else 'tts'

                logger.info(
                    f"Processing request for endpoint: {endpoint} with "
                    f"params: {params}"
                )

                if path == "tts" or not path:  # Handle root path as TTS
                    outname = f"tts-{int(time.time())}.wav"
                    output_path = await self.batch('tts', outname, params)
                elif path in ["clone", "clone_mul"]:
                    outname = f"clone-{int(time.time())}.wav"
                    output_path = await self.batch('clone', outname, params)
                elif path == "clone_eq":
                    outname = f"clone-eq-{int(time.time())}.wav"
                    output_path = await self.batch('clone_eq', outname, params)
                else:
                    return JSONResponse(
                        {"error": "Invalid endpoint"},
                        status_code=404
                    )

                # Read the file content
                with open(output_path, 'rb') as f:
                    audio_content = f.read()

                # Delete the output file immediately after reading
                try:
                    Path(output_path).unlink()
                    logger.info(
                        f"Deleted output file after sending: {output_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to delete output file {output_path}: {e}"
                    )
                    # Schedule for cleanup by the cleanup thread

                return Response(
                    audio_content,
                    media_type='audio/x-wav'
                )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(
                {
                    "error": str(e),
                    "message": (
                        "Please provide required parameters: text, and "
                        "optionally speaker/language or role"
                    )
                },
                status_code=400
            )

    def _setup_models(self) -> None:
        """Download and initialize TTS models on assigned device."""
        # Download models
        self._download_models()

        logger.info(f"Initializing models on device: {self.device}")

        try:
            # Initialize models with CUDA if available
            use_cuda = self.device.startswith('cuda')
            self.sft_model = CosyVoice(
                'pretrained_models/CosyVoice-300M-SFT',
                load_jit=use_cuda,
                load_trt=False,
                fp16=use_cuda
            )

            self.tts_model = CosyVoice2(
                'pretrained_models/CosyVoice2-0.5B',
                load_jit=use_cuda,
                load_trt=False,
                fp16=use_cuda
            )

            logger.info(f"Successfully initialized models on {self.device}")
        except Exception as e:
            logger.error(
                f"Failed to initialize models on {self.device}: {e}"
            )
            if self.device.startswith("cuda"):
                logger.info("Falling back to CPU")
                self.device = "cpu"
                # Initialize models without CUDA optimizations
                self.sft_model = CosyVoice(
                    'pretrained_models/CosyVoice-300M-SFT',
                    load_jit=False,
                    load_trt=False,
                    fp16=False
                )
                self.tts_model = CosyVoice2(
                    'pretrained_models/CosyVoice2-0.5B',
                    load_jit=False,
                    load_trt=False,
                    fp16=False
                )
            else:
                raise

    @staticmethod
    def _download_models() -> None:
        """Download required model files."""
        snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
        snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

    def _get_prompt_path(self, voice_type: Optional[str] = None) -> str:
        """Get the reference audio path for the specified voice type.

        Args:
            voice_type: The type of voice to use. If None, uses default voice.

        Returns:
            Path to the reference audio file.

        Raises:
            Exception: If the voice prompt file is not found.
        """
        if not voice_type:
            voice_type = self.default_voice
        if voice_type not in self.voice_prompts:
            logging.warning(
                f"Unknown voice type: {voice_type}, using {self.default_voice}"
            )
            voice_type = self.default_voice
        voice_path = Path(project_dir) / self.voice_prompts[voice_type]
        if not voice_path.exists():
            raise Exception(f"Voice prompt file not found: {voice_path}")
        return str(voice_path.resolve())

    def monitor_gpu_usage(self) -> None:
        """Monitor GPU memory usage for the current device."""
        if not hasattr(self, 'gpu_id') or self.gpu_id is None:
            return

        memory_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3

        logger.info(f"GPU {self.gpu_id} Usage:")
        logger.info(f"Allocated Memory: {memory_allocated:.2f}GB")
        logger.info(f"Reserved Memory: {memory_reserved:.2f}GB")

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and transforms input parameters.

        Args:
            params: Dictionary containing request parameters.

        Returns:
            Dictionary with validated and transformed parameters.

        Raises:
            ValueError: If required parameters are missing.
        """
        if not params.get('text'):
            raise ValueError("Missing required parameter: text")

        # Add default speed if not provided
        params['speed'] = float(params.get('speed', 1.0))

        return params

    async def batch(self, tts_type: str, outname: str, params: Dict[str, Any]) -> str:
        """Processes a batch text-to-speech request.

        Handles different types of TTS operations including standard synthesis,
        voice cloning, and cross-lingual synthesis. Manages audio file
        processing and model inference.

        Args:
            tts_type: Type of TTS operation ('tts', 'clone', or 'clone_eq').
            outname: Output filename for the generated audio.
            params: Request parameters including text and voice settings.

        Returns:
            Path to the generated audio file.

        Raises:
            Exception: If audio processing or synthesis fails.
        """
        if tts_type == 'tts':
            # Get voice type and reference audio path
            voice_type = params.get('voice_type')
            prompt_path = self._get_prompt_path(voice_type)

            # Process reference audio
            ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-ignore_unknown", "-y",
                        "-i", prompt_path,
                        "-ar", "16000", ref_audio
                    ],
                    check=True, capture_output=True, text=True
                )
                prompt_speech_16k = load_wav(ref_audio, 16000)
                # Delete reference audio file immediately after loading
                try:
                    Path(ref_audio).unlink()
                except Exception as del_e:
                    logger.warning(
                        f"Failed to delete reference audio file {ref_audio}: "
                        f"{del_e}"
                    )
            except Exception as e:
                # Try to clean up the file even if processing failed
                try:
                    Path(ref_audio).unlink()
                except Exception:
                    pass
                raise Exception(f'Failed to process reference audio: {e}')

            # Use CosyVoice2's cross-lingual synthesis
            audio_list = []
            for _, j in enumerate(
                self.tts_model.inference_cross_lingual(
                    params['text'], prompt_speech_16k, stream=False,
                    speed=params['speed']
                )
            ):
                audio_list.append(j['tts_speech'])

        elif tts_type == 'clone_eq' and params.get('reference_text'):
            if (
                not params['reference_audio']
                or not os.path.exists(params['reference_audio'])
            ):
                raise Exception(
                    f'Reference audio not found: {params["reference_audio"]}'
                )

            ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-ignore_unknown", "-y",
                        "-i", params['reference_audio'],
                        "-ar", "16000", ref_audio
                    ],
                    check=True, capture_output=True, text=True
                )
                prompt_speech_16k = load_wav(ref_audio, 16000)
                # Delete reference audio file immediately after loading
                try:
                    Path(ref_audio).unlink()
                except Exception as del_e:
                    logger.warning(
                        f"Failed to delete reference audio file {ref_audio}: "
                        f"{del_e}"
                    )
            except Exception as e:
                # Try to clean up the file even if processing failed
                try:
                    Path(ref_audio).unlink()
                except Exception:
                    pass
                raise Exception(f'Failed to process reference audio: {e}')

            audio_list = []
            for _, j in enumerate(
                self.tts_model.inference_zero_shot(
                    params['text'], params.get('reference_text'),
                    prompt_speech_16k, stream=False, speed=params['speed']
                )
            ):
                audio_list.append(j['tts_speech'])
        else:
            if (
                not params['reference_audio']
                or not os.path.exists(params['reference_audio'])
            ):
                raise Exception(
                    f'Reference audio not found: {params["reference_audio"]}'
                )

            ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-ignore_unknown", "-y",
                        "-i", params['reference_audio'],
                        "-ar", "16000", ref_audio
                    ],
                    check=True, capture_output=True, text=True
                )
                prompt_speech_16k = load_wav(ref_audio, 16000)
                # Delete reference audio file immediately after loading
                try:
                    Path(ref_audio).unlink()
                except Exception as del_e:
                    logger.warning(
                        f"Failed to delete reference audio file {ref_audio}: "
                        f"{del_e}"
                    )
            except Exception as e:
                # Try to clean up the file even if processing failed
                try:
                    Path(ref_audio).unlink()
                except Exception:
                    pass
                raise Exception(f'Failed to process reference audio: {e}')

            audio_list = []
            for _, j in enumerate(
                self.tts_model.inference_cross_lingual(
                    params['text'], prompt_speech_16k, stream=False,
                    speed=params['speed']
                )
            ):
                audio_list.append(j['tts_speech'])

        # Monitor GPU usage before inference
        if self.device.startswith('cuda'):
            logger.info("GPU usage before tensor operations:")
            self.monitor_gpu_usage()

        # Move audio tensors to the correct device and concatenate
        audio_data = torch.concat(
            [audio.to(self.device) for audio in audio_list],
            dim=1
        )

        # Monitor GPU usage after tensor operations
        if self.device.startswith('cuda'):
            logger.info("GPU usage after tensor operations:")
            self.monitor_gpu_usage()

        # Move to CPU for saving
        audio_data = audio_data.cpu()

        # Monitor GPU usage after moving to CPU
        if self.device.startswith('cuda'):
            logger.info("GPU usage after moving to CPU:")
            self.monitor_gpu_usage()
            # Clear CUDA cache
            torch.cuda.empty_cache()

        # Always use 24000 sample rate for CosyVoice2
        output_path = f"{tmp_dir}/{outname}"
        torchaudio.save(output_path, audio_data, 24000, format="wav")

        return output_path

def setup_logging() -> None:
    """Configure logging settings for the application.

    Sets up logging with both file and console handlers. Log files are created
    daily in the logs directory with timestamps in the filename.

    The logging configuration includes:
        - INFO level logging
        - Timestamp, logger name, log level, and message in the format
        - Daily rotating file handler
        - Console output handler
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{logs_dir}/{datetime.datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

def initialize_ray() -> None:
    """Initialize Ray connection in container environment.
    
    Connects to the Ray cluster using the RAY_ADDRESS environment variable.
    This function assumes Ray is already running in the container environment.

    Raises:
        RuntimeError: If RAY_ADDRESS environment variable is not set.
        Exception: If connection to Ray cluster fails.
    """
    ray_address = os.getenv('RAY_ADDRESS')
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS environment variable not set")

    try:
        ray.init(address=ray_address)
        logger.info(f"Connected to Ray cluster at {ray_address}")
    except Exception as e:
        logger.error(f"Failed to connect to Ray cluster: {e}")
        raise

def cleanup_resources() -> None:
    """Clean up resources when shutting down.
    
    Handles graceful shutdown of Ray services and GPU resources. This function
    performs the following cleanup tasks:
        1. Clears GPU memory if CUDA is available
        2. Shuts down Ray Serve deployments
        3. Shuts down Ray runtime

    Raises:
        SystemExit: With exit code 1 if cleanup fails.
    """
    try:
        if ray.is_initialized():
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory")

            # Shutdown Ray
            serve.shutdown()
            ray.shutdown()
            logger.info("Ray services stopped")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        sys.exit(1)

def register_signals() -> None:
    """Register signal handlers for graceful shutdown.
    
    Sets up handlers for SIGTERM and SIGINT signals to ensure graceful
    shutdown of the service. When these signals are received, the service:
        1. Logs the received signal
        2. Calls cleanup_resources() to perform cleanup
        3. Exits the process with status code 0
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        cleanup_resources()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def safe_delete_file(file_path: Path) -> bool:
    """Safely delete a file with retries on Unix systems.

    Attempts to delete a file multiple times with delays between attempts to
    handle temporary file system locks or other transient issues.

    Args:
        file_path: Path to the file to delete.

    Returns:
        bool: True if deletion was successful or file doesn't exist,
            False if deletion failed after all retries.

    Note:
        - Makes up to 3 attempts to delete the file
        - Waits 1 second between attempts
        - Logs success/failure at appropriate log levels
    """
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if not file_path.exists():
                return True

            file_path.unlink()
            logger.info(f"Successfully deleted file: {file_path}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to delete {file_path} (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to delete {file_path} after {max_retries} attempts: {e}")
                return False

    return False

def cleanup_old_files() -> None:
    """Clean up temporary audio files older than 48 hours.

    Scans the temporary directory for files older than 48 hours and attempts
    to delete them. Tracks and logs statistics about the cleanup operation
    including:
        - Number of files successfully deleted
        - Number of files that failed to delete
        - Total disk space freed
        - Any errors encountered during the process

    Note:
        - Only processes regular files (not directories)
        - Uses safe_delete_file() for reliable deletion
        - Continues processing even if individual files fail
        - Logs final cleanup statistics if any actions were taken
    """
    try:
        current_time = time.time()
        # 48 hours in seconds
        max_age = 48 * 60 * 60

        # Get all files in tmp directory
        deleted_count = 0
        failed_count = 0
        total_size_freed = 0

        for file_path in Path(tmp_dir).glob('*'):
            if not file_path.is_file():
                continue

            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age:
                    # Get file size before deletion
                    try:
                        file_size = file_path.stat().st_size
                    except Exception:
                        file_size = 0

                    # Attempt to delete the file
                    if safe_delete_file(file_path):
                        deleted_count += 1
                        total_size_freed += file_size
                    else:
                        failed_count += 1
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                failed_count += 1

        # Log cleanup results
        if deleted_count > 0 or failed_count > 0:
            logger.info(
                f"Cleanup completed: {deleted_count} files deleted "
                f"({total_size_freed / (1024*1024):.2f} MB freed), "
                f"{failed_count} files failed to delete"
            )

    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")

def schedule_cleanup() -> None:
    """Schedule periodic cleanup of old files.

    Runs in a separate thread to periodically clean up old temporary files.
    The cleanup process:
        - Runs every hour
        - Calls cleanup_old_files() to perform the actual cleanup
        - Continues running until the program exits
        - Logs any errors encountered during cleanup

    Note:
        This function is designed to run as a daemon thread and will be
        automatically terminated when the main program exits.
    """
    while True:
        try:
            cleanup_old_files()
        except Exception as e:
            logger.error(f"Error in cleanup schedule: {e}")
        # Sleep for 1 hour before next cleanup check
        time.sleep(3600)

def main() -> None:
    """Initialize and run the TTS service in container environment.
    
    This is the main entry point for the TTS service. It performs the following
    initialization steps:
        1. Sets up logging configuration
        2. Registers signal handlers for graceful shutdown
        3. Starts the file cleanup thread
        4. Initializes Ray connection
        5. Starts Ray Serve
        6. Deploys the CosyVoice service
        7. Maintains the service running state

    The service is configured to:
        - Listen on all interfaces (0.0.0.0)
        - Use port 9998
        - Deploy on every node in the cluster
        - Handle graceful shutdowns
        - Monitor GPU availability

    Raises:
        SystemExit: With exit code 1 if service startup fails.
    """
    setup_logging()
    register_signals()

    try:
        # Start cleanup thread for temporary files
        cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info("Started file cleanup scheduler")

        # Initialize Ray connection
        initialize_ray()

        # Start Ray Serve
        serve.start(
            http_options={
                "host": "0.0.0.0",
                "port": 9998,
                "location": "EveryNode"
            }
        )

        # Deploy service
        deployment = CosyVoiceService.bind()
        serve.run(
            deployment,
            route_prefix="/v1/model/cosyvoice",
            name="cosyvoice_service"
        )

        # Log GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Running with {gpu_count} GPU(s)")
        else:
            logger.info("Running on CPU")

        logger.info("Service started at http://localhost:9998")

        # Keep container running
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
