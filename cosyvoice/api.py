"""CosyVoice2 Text-to-Speech Service API.

This module provides a Ray Serve based HTTP API for text-to-speech synthesis
using the CosyVoice2 model. The API is designed for high-performance, scalable
deployment with GPU acceleration and efficient resource management.

Features:
1. Zero-shot voice cloning - Clone any voice with a single audio sample
2. Cross-lingual synthesis - Maintain voice identity across different languages
3. Instruction-based synthesis - Control speech characteristics with natural language
4. Fine-grained control - Adjust emotions, prosody, and speaking style
5. Streaming audio generation - Real-time audio synthesis for low-latency applications
6. Reference audio caching - Improved performance for repeated voice cloning

The implementation uses Ray Serve for deployment management and scaling,
with automatic GPU detection and allocation. It includes comprehensive
error handling, logging, and resource management.

The API exposes several HTTP endpoints:
- /v1/model/cosyvoice/tts: Standard text-to-speech with default voice
- /v1/model/cosyvoice/zero_shot: Zero-shot voice cloning
- /v1/model/cosyvoice/cross_lingual: Cross-lingual voice cloning
- /v1/model/cosyvoice/instruct: Instruction-based TTS
- /v1/model/cosyvoice/healthcheck: Health monitoring endpoint

Example usage:
    # Start the service
    python cosyvoice/api.py
    
    # Make a request to the API
    curl -X POST http://localhost:9998/v1/model/cosyvoice/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "voice_type": "qwen"}' \
        --output output.wav
        
    # Zero-shot voice cloning
    curl -X POST http://localhost:9998/v1/model/cosyvoice/zero_shot \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello world",
            "reference_audio": "/path/to/audio.wav",
            "reference_text": "Reference text"
        }' \
        --output output.wav
        
    # Streaming audio generation
    curl -X POST http://localhost:9998/v1/model/cosyvoice/tts \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello world",
            "voice_type": "qwen",
            "stream": true
        }' \
        --output output.wav
"""

import os
import sys
import json
import time
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Initialize CUDA before importing torch
import torch.backends.cuda
import torch.backends.cudnn

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

import ray
from ray import serve
import torch
import torchaudio
from modelscope import snapshot_download
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
import io
import wave

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Type aliases
PathLike = Union[str, Path]
JsonDict = Dict[str, Any]

# Constants
VOICE_PROMPTS = {
    "qwen": "asset/qwen.wav",  # Default qwen voice
}
DEFAULT_VOICE = "qwen"
SAMPLE_RATE = 24000  # CosyVoice2 uses 24kHz sample rate

# Setup paths
root_dir = Path(__file__).resolve().parent.absolute()
project_dir = root_dir.parent
tmp_dir = project_dir / 'tmp'
logs_dir = project_dir / 'logs'
asset_dir = project_dir / 'asset'

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(asset_dir, exist_ok=True)

# Convert paths to strings
root_dir = str(root_dir)
tmp_dir = str(tmp_dir)
logs_dir = str(logs_dir)
asset_dir = str(asset_dir)

# Configure logging
from logging.handlers import TimedRotatingFileHandler
import gzip
import shutil

class GZipRotator:
    """Rotator for log files that compresses rotated logs with gzip.
    
    This class is used with TimedRotatingFileHandler to compress
    rotated log files, saving disk space while maintaining log history.
    """
    
    def __call__(self, source, dest):
        """Compress the rotated log file with gzip.
        
        Args:
            source: Path to the source log file.
            dest: Path to the destination log file.
        """
        with open(source, 'rb') as f_in:
            with gzip.open(f"{dest}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configure file logger
log_file = os.path.join(logs_dir, 'cosyvoice.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler.setLevel(logging.INFO)

# Add file handler to root logger
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

# Check CUDA environment
if torch.cuda.is_available():
    # Force PyTorch to reinitialize CUDA state
    torch.cuda.init()
    torch.cuda.empty_cache()

    # Log CUDA information
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
    logger.warning("CUDA is not available. Running on CPU only.")

# Configure environment
os.environ['PATH'] = f"{root_dir}:{root_dir}/ffmpeg:" + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')

# Audio processing utility class
class AudioProcessor:
    """Audio processing utility class.

    This class provides static methods for common audio processing tasks
    such as resampling, normalization, and validation.
    """

    @staticmethod
    def process_reference_audio(
        audio_path: PathLike,
        output_path: PathLike,
        sample_rate: int = 16000
    ) -> None:
        """Processes a reference audio file using FFmpeg.

        Converts the input audio to a mono, 16-bit PCM WAV file at the
        specified sample rate.

        Args:
            audio_path: Path to the input audio file.
            output_path: Path to save the processed audio file.
            sample_rate: Target sample rate for the output audio.
                Defaults to 16000 Hz.

        Raises:
            FileNotFoundError: If the input audio file does not exist.
            subprocess.CalledProcessError: If FFmpeg processing fails.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(audio_path),
                    "-ar", str(sample_rate),
                    "-ac", "1",  # Convert to mono
                    "-acodec", "pcm_s16le",  # 16-bit PCM encoding
                    str(output_path)
                ],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg processing failed: {e.stderr}")
            raise

    @staticmethod
    def normalize_audio(audio_tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes the audio data to the range [-1, 1].

        Args:
            audio_tensor: Input audio tensor.

        Returns:
            Normalized audio tensor.
        """
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()

        max_val = audio_tensor.abs().max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val

        return audio_tensor

    @staticmethod
    def validate_audio(audio_tensor: torch.Tensor) -> bool:
        """Validates the audio data.

        Checks if the input is a tensor, not empty, and does not contain
        NaN or infinite values.

        Args:
            audio_tensor: Input audio tensor.

        Returns:
            True if the audio data is valid, False otherwise.
        """
        if not torch.is_tensor(audio_tensor):
            return False

        if audio_tensor.numel() == 0:
            return False

        if audio_tensor.isnan().any():
            return False

        if audio_tensor.isinf().any():
            return False

        return True

@serve.deployment(
    max_ongoing_requests=1,  # Process one request at a time
    health_check_period_s=30,  # Regular health checks
    health_check_timeout_s=60,  # Health check timeout
    graceful_shutdown_timeout_s=120  # Allow time for graceful shutdown
)
class CosyVoiceService:
    """CosyVoice2 service class.

    This class implements a Ray Serve deployment for the CosyVoice2
    text-to-speech model. It handles model loading, GPU management,
    request processing, and audio processing.
    """

    def __init__(self) -> None:
        """Initializes the CosyVoiceService.

        - Detects and configures GPU usage if available.
        - Sets up the CosyVoice2 model.
        - Initializes the audio processor.
        - Configures voice prompts.
        - Registers a cleanup handler for resource management.
        """
        # Cache for processed reference audio
        self._reference_audio_cache = {}
        if torch.cuda.is_available():
            try:
                # Get GPU IDs assigned by Ray
                gpu_ids = ray.get_gpu_ids()
                if gpu_ids:
                    # Get worker info
                    worker_id = ray.get_runtime_context().worker.worker_id
                    # Use first assigned GPU
                    gpu_id = str(gpu_ids[0])

                    # Get CUDA_VISIBLE_DEVICES
                    cuda_visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if cuda_visible_str and cuda_visible_str != "NoDevFiles":
                        cuda_visible_list = cuda_visible_str.split(",")
                        try:
                            # Map Ray GPU ID to CUDA device ID
                            device_id = cuda_visible_list.index(gpu_id)
                            logger.info(f"Worker {worker_id} using GPU {gpu_id} (CUDA device {device_id})")

                            # Set device
                            torch.cuda.set_device(device_id)
                            self.device = f"cuda:{device_id}"
                            self.gpu_id = device_id

                            # Test GPU access
                            test_tensor = torch.tensor([1.0], device=self.device)
                            del test_tensor

                            # Log GPU memory info
                            logger.info(f"GPU {device_id} memory allocated: {torch.cuda.memory_allocated(device_id) / 1024**2:.1f}MB")
                            logger.info(f"GPU {device_id} memory cached: {torch.cuda.memory_reserved(device_id) / 1024**2:.1f}MB")
                        except ValueError:
                            logger.error(f"GPU {gpu_id} not found in CUDA_VISIBLE_DEVICES={cuda_visible_str}")
                            raise RuntimeError(f"GPU {gpu_id} not found in CUDA_VISIBLE_DEVICES")
                    else:
                        logger.warning("CUDA_VISIBLE_DEVICES not set, falling back to CPU")
                        self.device = "cpu"
                        self.gpu_id = None
                else:
                    logger.warning("No GPU assigned by Ray, falling back to CPU")
                    self.device = "cpu"
                    self.gpu_id = None
            except Exception as e:
                logger.error(f"Error setting GPU device: {e}")
                logger.warning("Falling back to CPU")
                self.device = "cpu"
                self.gpu_id = None
                # Clear any GPU memory that might have been allocated
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"
            self.gpu_id = None

        self._setup_models()
        self.audio_processor = AudioProcessor()
        self.voice_prompts = VOICE_PROMPTS
        self.default_voice = DEFAULT_VOICE

        # Register cleanup handler using Ray's public API
        try:
            import atexit
            atexit.register(self._cleanup_resources)
        except Exception as e:
            logger.error(f"Failed to register cleanup handler: {e}")


    def _cleanup_resources(self) -> None:
        """Cleans up resources used by the deployment.

        - Clears GPU memory if CUDA is used.
        - Deletes model references to free up memory.
        """
        try:
            # Clear GPU memory if using CUDA
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                try:
                    # Move models to CPU before clearing GPU memory
                    if hasattr(self, 'model'):
                        self.model = self.model.cpu()
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared GPU {self.gpu_id} memory")
                except Exception as e:
                    logger.error(f"Error clearing GPU memory: {e}")

            # Delete model references
            if hasattr(self, 'model'):
                del self.model

            logger.info("Cleaned up deployment resources")
        except Exception as e:
            logger.error(f"Error during deployment cleanup: {e}")

    def _setup_models(self):
        """Initializes and loads the CosyVoice2 models.

        Downloads the model if necessary and initializes it on the
        appropriate device (CPU or GPU). Handles fallback to CPU if GPU
        initialization fails.
        """
        self._download_models()
        logger.info(f"Initializing models on device: {self.device}")

        try:
            # Initialize models with CUDA if available
            use_cuda = self.device.startswith('cuda')
            self.model = CosyVoice2(
                'pretrained_models/CosyVoice2-0.5B',
                load_jit=use_cuda,
                load_trt=False,
                fp16=use_cuda
            )
            logger.info(f"Successfully initialized models on {self.device}")

            # Log GPU memory usage after model initialization
            if use_cuda:
                logger.info(f"GPU memory after model init: {torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to initialize models on {self.device}: {e}")
            if self.device.startswith("cuda"):
                logger.info("Falling back to CPU")
                self.device = "cpu"
                # Initialize models without CUDA optimizations
                self.model = CosyVoice2(
                    'pretrained_models/CosyVoice2-0.5B',
                    load_jit=False,
                    load_trt=False,
                    fp16=False
                )
            else:
                raise

    @staticmethod
    def _download_models():
        """Downloads the CosyVoice2 model files from ModelScope."""
        snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

    async def __call__(self, request: Request) -> Response:
        """Handles incoming HTTP requests.

        Processes POST requests for text-to-speech synthesis, supporting
        different endpoints for various synthesis modes.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A Starlette Response object containing the synthesized audio
            or an error message.

        Raises:
            Exception: If any error occurs during request processing.
        """
        # Handle CORS preflight request
        if request.method == "OPTIONS":
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
            return Response("", headers=headers)

        # Add CORS headers to all responses
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        try:
            if request.method == "POST":
                if request.headers.get("content-type") == "application/json":
                    params = await request.json()
                else:
                    form = await request.form()
                    params = dict(form)

                path = request.url.path.strip('/')
                parts = path.split('/')
                endpoint = parts[-1] if parts else 'tts'  # Default to 'tts' if no endpoint

                logger.info(f"Processing request: {endpoint}, parameters: {params}")

                if endpoint == "healthcheck":
                    return JSONResponse({"status": "healthy"})

                if endpoint not in ["tts", "zero_shot", "cross_lingual", "instruct"]:
                    return JSONResponse(
                        {"error": "Invalid endpoint"},
                        status_code=404
                    )

                # Validate params for TTS endpoints
                params = self._validate_params(params)

                # Check if streaming is requested
                stream_output = params.get('stream', False)

                if stream_output:
                    # Get voice prompt
                    voice_type = params.get('voice_type')
                    _, prompt_speech_16k = self._get_prompt_path(voice_type)

                    # Create a generator for streaming audio chunks
                    async def audio_stream():
                        # First chunk: WAV header
                        wav_buffer = io.BytesIO()
                        wav = wave.Wave_write(wav_buffer)
                        wav.setnchannels(1)  # Mono
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(SAMPLE_RATE)
                        wav.setnframes(0)  # We don't know total frames yet
                        wav.close()
                        yield wav_buffer.getvalue()

                        # Stream each audio chunk as it's generated
                        try:
                            # Use the cross-lingual inference mode with streaming enabled
                            # This generates audio incrementally as chunks become available
                            for _, j in enumerate(
                                self.model.inference_cross_lingual(
                                    params['text'], prompt_speech_16k, stream=True,
                                    speed=params['speed']
                                )
                            ):
                                # Extract audio chunk from model output
                                if 'tts_speech' in j and j['tts_speech'] is not None:
                                    audio = j['tts_speech']
                                    if audio is not None and audio.numel() > 0:
                                        # Normalize audio to [-1, 1] range
                                        audio = self.audio_processor.normalize_audio(audio)
                                        # Move tensor to CPU for further processing
                                        audio = audio.cpu()
                                        # Convert float32 [-1, 1] to int16 PCM format
                                        # Scale by 32767 to use full 16-bit range
                                        audio_int16 = (audio * 32767).to(torch.int16)
                                        # Convert to bytes and yield to client
                                        yield audio_int16.numpy().tobytes()
                        except Exception as e:
                            logger.error(f"Error during streaming: {e}")
                            raise

                    return StreamingResponse(
                        audio_stream(),
                        media_type='audio/wav',
                        headers=headers
                    )
                else:
                    # Generate audio based on endpoint
                    outname = f"{endpoint}-{int(time.time())}.wav"
                    output_path = await self.batch(endpoint, outname, params)

                    with open(output_path, 'rb') as f:
                        audio_content = f.read()

                    try:
                        Path(output_path).unlink()  # Delete the temporary file
                    except Exception as e:
                        logger.warning(f"Failed to delete output file {output_path}: {e}")

                    return Response(
                        audio_content,
                        media_type='audio/x-wav',
                        headers=headers
                    )

        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return JSONResponse(
                {"error": str(e)},
                status_code=400,
                headers=headers
            )

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the parameters received in the request.

        Ensures that required parameters are present and have valid values.

        Args:
            params: A dictionary of request parameters.

        Returns:
            The validated parameters.

        Raises:
            ValueError: If any required parameter is missing or invalid.
        """
        if not params.get('text'):
            raise ValueError("Missing required parameter: text")
        params['speed'] = float(params.get('speed', 1.0))  # Ensure speed is a float
        return params

    def _get_prompt_path(self, voice_type: Optional[str] = None) -> Tuple[str, torch.Tensor]:
        """Gets the path to the voice prompt audio file and its processed tensor.

        Args:
            voice_type: The type of voice to use (e.g., "qwen").

        Returns:
            A tuple of (file path, processed audio tensor).

        Raises:
            Exception: If the specified voice type is unknown or the
                prompt file is not found.
        """
        if not voice_type:
            voice_type = self.default_voice  # Use default if not provided
        if voice_type not in self.voice_prompts:
            logger.warning(f"Unknown voice type: {voice_type}, using default voice")
            voice_type = self.default_voice
        voice_path = Path(project_dir) / self.voice_prompts[voice_type]
        if not voice_path.exists():
            raise Exception(f"Voice prompt file not found: {voice_path}")

        path = str(voice_path.resolve())

        # Check cache first
        if path in self._reference_audio_cache:
            return path, self._reference_audio_cache[path]

        # Process and cache if not found
        ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
        try:
            self.audio_processor.process_reference_audio(path, ref_audio)
            prompt_speech_16k = load_wav(ref_audio, 16000)
            Path(ref_audio).unlink()

            # Cache the processed audio
            self._reference_audio_cache[path] = prompt_speech_16k
            return path, prompt_speech_16k
        except Exception as e:
            logger.error(f"Reference audio processing failed: {e}")
            if os.path.exists(ref_audio):
                Path(ref_audio).unlink()
            raise

    async def batch(self, tts_type: str, outname: str, params: Dict[str, Any]) -> str:
        """Performs batch text-to-speech processing.

        Handles different TTS modes (standard, zero-shot, cross-lingual,
        instruction-based) based on the `tts_type` parameter.

        Args:
            tts_type: The type of TTS operation to perform.
            outname: The name of the output audio file.
            params: A dictionary of parameters for the TTS operation.

        Returns:
            The path to the generated audio file.

        Raises:
            Exception: If any error occurs during audio generation or processing.
        """
        try:
            # Get reference audio (now uses caching)
            if tts_type == 'tts':
                # Use default qwen.wav as reference audio
                voice_type = params.get('voice_type')
                _, prompt_speech_16k = self._get_prompt_path(voice_type)

                # Check if streaming is requested
                stream_output = params.get('stream', False)
                if stream_output:
                    # Handle streaming request by collecting all chunks first
                    audio_list = []
                    for _, j in enumerate(
                        self.model.inference_cross_lingual(
                            params['text'], prompt_speech_16k, stream=True,
                            speed=params['speed']
                        )
                    ):
                        if 'tts_speech' in j and j['tts_speech'] is not None:
                            audio = j['tts_speech']
                            if audio is not None and audio.numel() > 0:  # Less strict validation
                                audio = self.audio_processor.normalize_audio(audio)
                                audio = audio.cpu()
                                audio_list.append(audio)

                    if not audio_list:
                        raise Exception("Failed to generate valid audio")

                    # Concatenate audio segments
                    audio_data = torch.cat(audio_list, dim=1)

                    # Save to memory buffer
                    buffer = io.BytesIO()
                    torchaudio.save(
                        buffer,
                        audio_data,
                        SAMPLE_RATE,
                        format="wav",
                        encoding='PCM_S',
                        bits_per_sample=16
                    )
                    buffer.seek(0)

                    # Save to temporary file
                    output_path = f"{tmp_dir}/{outname}"
                    with open(output_path, 'wb') as f:
                        f.write(buffer.read())
                    return output_path

                else:
                    # Handle non-streaming request
                    audio_list = []
                    for _, j in enumerate(
                        self.model.inference_cross_lingual(
                            params['text'], prompt_speech_16k, stream=False,
                            speed=params['speed']
                        )
                    ):
                        if 'tts_speech' in j and j['tts_speech'] is not None:
                            audio = j['tts_speech']
                            if audio is not None and audio.numel() > 0:  # Less strict validation
                                audio = self.audio_processor.normalize_audio(audio)
                                audio = audio.cpu()
                                audio_list.append(audio)

                    if not audio_list:
                        raise Exception("Failed to generate valid audio")

                    # Concatenate audio segments
                    audio_data = torch.cat(audio_list, dim=1)

                    # Save audio to file
                    output_path = f"{tmp_dir}/{outname}"
                    torchaudio.save(
                        output_path,
                        audio_data,
                        SAMPLE_RATE,
                        encoding='PCM_S',
                        bits_per_sample=16
                    )

                    # Validate file
                    if not os.path.exists(output_path):
                        raise Exception("Audio file failed to save")

                    file_size = os.path.getsize(output_path)
                    logger.info(f"Generated audio file size: {file_size/1024:.1f}KB")

                    if file_size < 1024:  # Files smaller than 1KB may be problematic
                        raise Exception("Generated audio file is too small, may be invalid")

                    return output_path

            elif tts_type == 'zero_shot':
                # Use user-provided reference audio and text
                if not params.get('reference_text'):
                    raise Exception("Missing required parameter: reference_text")
                if not params.get('reference_audio') or not os.path.exists(params['reference_audio']):
                    raise Exception(f"Reference audio not found: {params['reference_audio']}")

                # Check cache for user reference audio
                ref_path = str(Path(params['reference_audio']).resolve())
                if ref_path in self._reference_audio_cache:
                    prompt_speech_16k = self._reference_audio_cache[ref_path]
                else:
                    ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
                    try:
                        self.audio_processor.process_reference_audio(params['reference_audio'], ref_audio)
                        prompt_speech_16k = load_wav(ref_audio, 16000)
                        Path(ref_audio).unlink()
                        # Cache the processed audio
                        self._reference_audio_cache[ref_path] = prompt_speech_16k
                    except Exception as e:
                        logger.error(f"Reference audio processing failed: {e}")
                        if os.path.exists(ref_audio):
                            Path(ref_audio).unlink()
                        raise

                # Process audio with incremental generation
                audio_list = []
                for _, j in enumerate(
                    self.model.inference_zero_shot(
                        params['text'], params['reference_text'],
                        prompt_speech_16k, stream=False, speed=params['speed']
                    )
                ):
                    if 'tts_speech' in j and j['tts_speech'] is not None:
                        audio = j['tts_speech']
                        if self.audio_processor.validate_audio(audio):
                            audio = self.audio_processor.normalize_audio(audio)
                            audio = audio.cpu()
                            audio_list.append(audio)
                        else:
                            logger.warning("Skipping invalid audio segment")

                if not audio_list:
                    raise Exception("Failed to generate valid audio")

                # Concatenate audio segments
                audio_data = torch.cat(audio_list, dim=1)

                # Save audio to file
                output_path = f"{tmp_dir}/{outname}"
                torchaudio.save(
                    output_path,
                    audio_data,
                    SAMPLE_RATE,
                    encoding='PCM_S',
                    bits_per_sample=16
                )

                # Validate file
                if not os.path.exists(output_path):
                    raise Exception("Audio file failed to save")

                file_size = os.path.getsize(output_path)
                logger.info(f"Generated audio file size: {file_size/1024:.1f}KB")

                if file_size < 1024:  # Files smaller than 1KB may be problematic
                    raise Exception("Generated audio file is too small, may be invalid")

                return output_path

            elif tts_type == 'cross_lingual':
                # Use user-provided reference audio
                if not params.get('reference_audio') or not os.path.exists(params['reference_audio']):
                    raise Exception(f"Reference audio not found: {params['reference_audio']}")

                # Check cache for user reference audio
                ref_path = str(Path(params['reference_audio']).resolve())
                if ref_path in self._reference_audio_cache:
                    prompt_speech_16k = self._reference_audio_cache[ref_path]
                else:
                    ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
                    try:
                        self.audio_processor.process_reference_audio(params['reference_audio'], ref_audio)
                        prompt_speech_16k = load_wav(ref_audio, 16000)
                        Path(ref_audio).unlink()
                        # Cache the processed audio
                        self._reference_audio_cache[ref_path] = prompt_speech_16k
                    except Exception as e:
                        logger.error(f"Reference audio processing failed: {e}")
                        if os.path.exists(ref_audio):
                            Path(ref_audio).unlink()
                        raise

                # Process audio with incremental generation
                audio_list = []
                for _, j in enumerate(
                    self.model.inference_cross_lingual(
                        params['text'], prompt_speech_16k, stream=False,
                        speed=params['speed']
                    )
                ):
                    if 'tts_speech' in j and j['tts_speech'] is not None:
                        audio = j['tts_speech']
                        if self.audio_processor.validate_audio(audio):
                            audio = self.audio_processor.normalize_audio(audio)
                            audio = audio.cpu()
                            audio_list.append(audio)
                        else:
                            logger.warning("Skipping invalid audio segment")

                if not audio_list:
                    raise Exception("Failed to generate valid audio")

                # Concatenate audio segments
                audio_data = torch.cat(audio_list, dim=1)

                # Save audio to file
                output_path = f"{tmp_dir}/{outname}"
                torchaudio.save(
                    output_path,
                    audio_data,
                    SAMPLE_RATE,
                    encoding='PCM_S',
                    bits_per_sample=16
                )

                # Validate file
                if not os.path.exists(output_path):
                    raise Exception("Audio file failed to save")

                file_size = os.path.getsize(output_path)
                logger.info(f"Generated audio file size: {file_size/1024:.1f}KB")

                if file_size < 1024:  # Files smaller than 1KB may be problematic
                    raise Exception("Generated audio file is too small, may be invalid")

                return output_path

            else:  # instruct
                # Use instruction control
                if not params.get('instruction'):
                    raise Exception("Missing required parameter: instruction")

                # Get reference audio (now uses caching)
                voice_type = params.get('voice_type')
                _, prompt_speech_16k = self._get_prompt_path(voice_type)

                # Process audio with incremental generation
                audio_list = []
                for _, j in enumerate(
                    self.model.inference_instruct2(
                        params['text'],
                        params['instruction'],
                        prompt_speech_16k,
                        stream=False
                    )
                ):
                    if 'tts_speech' in j and j['tts_speech'] is not None:
                        audio = j['tts_speech']
                        if self.audio_processor.validate_audio(audio):
                            audio = self.audio_processor.normalize_audio(audio)
                            audio = audio.cpu()
                            audio_list.append(audio)
                        else:
                            logger.warning("Skipping invalid audio segment")

                if not audio_list:
                    raise Exception("Failed to generate valid audio")

                # Concatenate audio segments
                audio_data = torch.cat(audio_list, dim=1)

                # Save audio to file
                output_path = f"{tmp_dir}/{outname}"
                torchaudio.save(
                    output_path,
                    audio_data,
                    SAMPLE_RATE,
                    encoding='PCM_S',
                    bits_per_sample=16
                )

                # Validate file
                if not os.path.exists(output_path):
                    raise Exception("Audio file failed to save")

                file_size = os.path.getsize(output_path)
                logger.info(f"Generated audio file size: {file_size/1024:.1f}KB")

                if file_size < 1024:  # Files smaller than 1KB may be problematic
                    raise Exception("Generated audio file is too small, may be invalid")

                return output_path

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise

def main():
    """Main function to start the CosyVoice service.
    
    This function:
    1. Detects available GPUs
    2. Connects to a Ray cluster (or starts one if not available)
    3. Starts the Ray Serve service
    4. Deploys the CosyVoiceService with appropriate resources
    5. Sets up routing for the API endpoints
    
    The service is deployed with one replica per available GPU,
    with each replica getting one dedicated GPU. If no GPUs are
    available, the function will raise an error.
    
    Raises:
        RuntimeError: If no GPUs are available.
        Exception: If any error occurs during service startup.
    """
    try:
        # Get GPU count
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} available GPU(s)")

        # Set CUDA device order
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        # Verify GPU availability
        if gpu_count == 0:
            raise RuntimeError("No GPUs available")

        # Connect to existing Ray cluster
        if not ray.is_initialized():
            ray.init(
                address="auto",  # Connect to existing cluster
                namespace="serve",  # Use serve namespace
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_LAUNCH_BLOCKING": "1"
                    }
                }
            )
            logger.info("Connected to existing Ray cluster")

            # Log available resources
            cluster_resources = ray.cluster_resources()
            logger.info(f"Cluster resources: {cluster_resources}")

            # Check if cluster has enough GPUs
            available_gpus = int(cluster_resources.get('GPU', 0))
            if available_gpus < gpu_count:
                logger.warning(f"Cluster only has {available_gpus} GPUs, reducing replicas from {gpu_count}")
                gpu_count = available_gpus

        # Start service
        serve.start(
            detached=True,
            http_options={
                "host": "0.0.0.0",
                "port": 9998,
                "location": "HeadOnly"
            }
        )

        # Deploy service
        logger.info(f"Deploying CosyVoice Service with {gpu_count} replica(s)")

        # Create deployment with GPU resources
        deployment = CosyVoiceService.options(
            num_replicas=gpu_count,
            ray_actor_options={
                "num_cpus": 1,
                "num_gpus": 1,  # Each replica gets one GPU
                "memory": 4 * 1024 * 1024 * 1024,  # 4GB RAM per replica
            }
        ).bind()

        # Run service
        serve.run(
            deployment,
            route_prefix="/v1/model/cosyvoice",
            name="cosyvoice_service"
        )

        # Wait for deployment to complete
        time.sleep(5)

        logger.info(f"Service running. API available at: /v1/model/cosyvoice")

    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
