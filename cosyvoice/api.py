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
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure CUDA environment variables for optimal GPU performance
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ensures consistent GPU device ordering
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enables better error reporting for CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Prevents memory fragmentation

import ray
from ray import serve
import torch
import torchaudio
from modelscope import snapshot_download
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, StreamingResponse
import io
import asyncio
import tempfile

# Import CosyVoice components
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Type aliases for improved code readability
PathLike = Union[str, Path]  # Represents file system paths
JsonDict = Dict[str, Any]    # Represents JSON-like dictionary structures

# Configuration constants
VOICE_PROMPTS = {"qwen": "asset/qwen.wav"}  # Available voice prompts
DEFAULT_VOICE = "qwen"  # Default voice type if none specified
SAMPLE_RATE = 24000  # Audio sample rate in Hz

# Set up directory structure
root_dir = Path(__file__).resolve().parent.absolute()
project_dir = root_dir.parent
tmp_dir = project_dir / 'tmp'  # Temporary file storage
logs_dir = project_dir / 'logs'  # Log storage
asset_dir = project_dir / 'asset'  # Voice prompt storage

# Create directories if they don't exist
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(asset_dir, exist_ok=True)

# Convert Path objects to strings for compatibility with libraries expecting string paths
root_dir = str(root_dir)
tmp_dir = str(tmp_dir)
logs_dir = str(logs_dir)
asset_dir = str(asset_dir)

# Configure logging system with console and file handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add rotating file logger
log_file = os.path.join(logs_dir, 'cosyvoice.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

# Initialize CUDA if available and configure PyTorch settings
if torch.cuda.is_available():
    torch.cuda.init()  # Initialize CUDA context
    torch.cuda.empty_cache()  # Clear any existing allocations
    
    # Configure PyTorch CUDA settings for optimal performance
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for faster matrix operations on Ampere GPUs
    torch.backends.cudnn.enabled = True  # Enable cuDNN for optimized operations
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for optimal performance
    torch.backends.cudnn.deterministic = False  # Disable deterministic mode for better performance
    
    # Log GPU information for diagnostics
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"CUDA Device {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    # Set default device to first GPU
    if torch.cuda.device_count() > 0:
        torch.cuda.set_device(0)
        logger.info(f"Default CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA is not available. Running on CPU only.")

# Update environment paths
os.environ['PATH'] = f"{root_dir}:{root_dir}/ffmpeg:" + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')


class AudioProcessor:
    """
    Utility class for audio processing operations.
    
    This class provides methods for processing audio files and tensors,
    including conversion, normalization, validation, and saving.
    All methods are implemented as static methods for easy use without
    instantiation.
    """
    
    @staticmethod
    def process_reference_audio(
        audio_path: PathLike,
        output_path: PathLike,
        sample_rate: int = 16000
    ) -> None:
        """
        Process a reference audio file to prepare it for voice cloning.
        
        Converts the audio file to a standardized format using FFmpeg:
        - Single channel (mono)
        - 16-bit PCM encoding
        - Specified sample rate (default: 16kHz)
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path where the processed audio will be saved
            sample_rate: Target sample rate in Hz (default: 16000)
            
        Raises:
            FileNotFoundError: If the input audio file doesn't exist
            subprocess.CalledProcessError: If FFmpeg processing fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",  # Overwrite output if it exists
                    "-i", str(audio_path),  # Input file
                    "-ar", str(sample_rate),  # Set sample rate
                    "-ac", "1",  # Convert to mono
                    "-acodec", "pcm_s16le",  # 16-bit PCM encoding
                    str(output_path)  # Output file
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
        """
        Normalize audio tensor to the range [-1, 1].
        
        This is important for consistent audio quality and to prevent clipping.
        
        Args:
            audio_tensor: Input audio tensor
            
        Returns:
            Normalized audio tensor with values scaled to [-1, 1]
        """
        # Ensure audio is 2D (batch_size, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Convert to float32 if needed
        if audio_tensor.dtype != torch.float32:
            audio_tensor = audio_tensor.float()

        # Normalize to [-1, 1] range
        max_val = audio_tensor.abs().max()
        if max_val > 0:
            audio_tensor = audio_tensor / max_val

        return audio_tensor

    @staticmethod
    def validate_audio(audio_tensor: torch.Tensor) -> bool:
        """
        Validate audio tensor to ensure it's usable for further processing.
        
        Performs checks for common issues that could cause problems in audio processing.
        
        Args:
            audio_tensor: Audio tensor to validate
            
        Returns:
            True if the audio tensor is valid, False otherwise
        """
        # Check if it's a tensor
        if not torch.is_tensor(audio_tensor):
            return False
        
        # Check if it's empty
        if audio_tensor.numel() == 0:
            return False
            
        # Check for NaN values
        if audio_tensor.isnan().any():
            return False
            
        # Check for infinite values
        if audio_tensor.isinf().any():
            return False
            
        return True

    @staticmethod
    async def save_audio(audio_data: torch.Tensor, output_path: str) -> str:
        """
        Save audio tensor to an MP3 file.
        
        This method first saves the audio as a WAV file, then converts it to MP3
        using FFmpeg for better compression, and finally performs validation
        to ensure the file was created correctly.
        
        Args:
            audio_data: Audio tensor to save
            output_path: Path where the MP3 file will be saved
            
        Returns:
            Path to the saved MP3 file
            
        Raises:
            Exception: If audio file fails to save or validate
        """
        temp_wav_path = f"{tmp_dir}/temp-{int(time.time())}.wav"
        
        try:
            # Save as WAV first (better quality for intermediate format)
            torchaudio.save(
                temp_wav_path,
                audio_data,
                SAMPLE_RATE,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            # Convert WAV to MP3 using FFmpeg
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_wav_path,
                "-codec:a", "libmp3lame",  # Use LAME MP3 encoder
                "-qscale:a", "2",  # Quality setting (2 is high quality)
                output_path
            ], check=True, capture_output=True)
            
            # Validate file was created
            if not os.path.exists(output_path):
                raise Exception("Audio file failed to save")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            logger.info(f"Generated MP3 file size: {file_size/1024:.1f}KB")
            
            # Files smaller than 1KB are likely invalid or empty
            if file_size < 1024:
                raise Exception("Generated audio file is too small, may be invalid")
            
            return output_path
        finally:
            # Clean up temporary WAV file
            if os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary WAV file: {e}")


@serve.deployment(
    max_ongoing_requests=1,    # Process one request at a time to avoid GPU memory issues
    health_check_period_s=30,  # Perform health checks every 30 seconds
    health_check_timeout_s=60, # Health check timeout after 60 seconds
    graceful_shutdown_timeout_s=120  # Allow 120 seconds for graceful shutdown
)
class CosyVoiceService:
    """
    Ray Serve deployment class for CosyVoice TTS service.
    
    This class handles:
    - Model initialization and GPU resource management
    - HTTP request processing for various TTS endpoints
    - Audio generation in both streaming and non-streaming modes
    - Reference audio caching for improved performance
    - Error handling and resource cleanup
    
    The service exposes several endpoints:
    - /tts: Standard TTS with default or specified voice
    - /zero_shot: Zero-shot voice cloning with reference audio
    - /cross_lingual: Cross-lingual voice cloning
    - /instruct: Instruction-based TTS for fine-grained control
    - /healthcheck: Service health monitoring
    """

    def __init__(self) -> None:
        """
        Initialize the CosyVoice service.
        
        Sets up GPU detection, model initialization, audio processing,
        reference audio caching, and resource cleanup handlers.
        """
        # Cache for processed reference audio to avoid redundant processing
        self._reference_audio_cache = {}
        
        # Set up GPU device if available, otherwise use CPU
        self.device, self.gpu_id = self._setup_device()
        
        # Initialize CosyVoice model
        self._setup_models()
        
        # Create audio processor instance
        self.audio_processor = AudioProcessor()
        
        # Set up voice prompts and default voice
        self.voice_prompts = VOICE_PROMPTS
        self.default_voice = DEFAULT_VOICE
        
        # Register cleanup handler to ensure proper resource release
        try:
            import atexit
            atexit.register(self._cleanup_resources)
        except Exception as e:
            logger.error(f"Failed to register cleanup handler: {e}")

    def _setup_device(self) -> Tuple[str, Optional[int]]:
        """
        Set up the device (GPU/CPU) for model execution.
        
        Attempts to use GPU if available, with proper handling of Ray's
        GPU allocation. Falls back to CPU if GPU is not available or
        if configuration issues are encountered.
        
        Returns:
            Tuple containing (device_string, gpu_id)
            - device_string: PyTorch device string (e.g., 'cuda:0' or 'cpu')
            - gpu_id: GPU device ID or None if using CPU
        """
        # If CUDA is not available, use CPU
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu", None
            
        try:
            # Get GPU IDs assigned by Ray
            gpu_ids = ray.get_gpu_ids()
            if not gpu_ids:
                logger.warning("No GPU assigned by Ray, falling back to CPU")
                return "cpu", None
                
            # Get worker info for logging
            worker_id = ray.get_runtime_context().worker.worker_id
            
            # Use first assigned GPU
            gpu_id = str(gpu_ids[0])

            # Get CUDA_VISIBLE_DEVICES environment variable
            cuda_visible_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible_str and cuda_visible_str != "NoDevFiles":
                cuda_visible_list = cuda_visible_str.split(",")
                try:
                    # Map Ray GPU ID to CUDA device ID
                    device_id = cuda_visible_list.index(gpu_id)
                    logger.info(f"Worker {worker_id} using GPU {gpu_id} (CUDA device {device_id})")

                    # Set PyTorch device
                    torch.cuda.set_device(device_id)
                    
                    # Test GPU access with a small tensor
                    test_tensor = torch.tensor([1.0], device=f"cuda:{device_id}")
                    del test_tensor

                    logger.info(f"GPU {device_id} memory allocated: {torch.cuda.memory_allocated(device_id) / 1024**2:.1f}MB")
                    return f"cuda:{device_id}", device_id
                except ValueError:
                    logger.error(f"GPU {gpu_id} not found in CUDA_VISIBLE_DEVICES={cuda_visible_str}")
            
            logger.warning("Falling back to CPU due to GPU configuration issues")
            return "cpu", None
        except Exception as e:
            logger.error(f"Error setting GPU device: {e}")
            logger.warning("Falling back to CPU")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "cpu", None

    def _cleanup_resources(self) -> None:
        """
        Clean up resources when the service is shutting down.
        
        This method ensures proper resource deallocation, including:
        - Moving models from GPU to CPU to free GPU memory
        - Clearing CUDA cache
        - Deleting model references
        
        This helps prevent memory leaks and ensures clean shutdown.
        """
        try:
            # Clean up GPU resources if using CUDA
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                try:
                    # Move models to CPU before clearing GPU memory
                    if hasattr(self, 'model'):
                        self.model = self.model.cpu()
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared GPU {self.gpu_id} memory")
                except Exception as e:
                    logger.error(f"Error clearing GPU memory: {e}")

            # Delete model references to free memory
            if hasattr(self, 'model'):
                del self.model

            logger.info("Cleaned up deployment resources")
        except Exception as e:
            logger.error(f"Error during deployment cleanup: {e}")

    def _setup_models(self):
        """
        Download and initialize the CosyVoice2 model.
        
        This method:
        - Downloads model files if not already present
        - Initializes the model with appropriate settings for the current device
        - Handles fallback to CPU if GPU initialization fails
        - Logs memory usage and initialization status
        """
        # Download model files
        self._download_models()
        logger.info(f"Initializing models on device: {self.device}")

        try:
            # Set optimization flags based on device
            use_cuda = self.device.startswith('cuda')
            
            # Initialize model with appropriate settings
            self.model = CosyVoice2(
                'pretrained_models/CosyVoice2-0.5B',
                load_jit=use_cuda,  # Use JIT compilation for CUDA
                load_trt=False,     # TensorRT not used by default
                fp16=use_cuda       # Use FP16 precision for CUDA
            )
            logger.info(f"Successfully initialized models on {self.device}")

            # Log GPU memory usage after model initialization
            if use_cuda:
                logger.info(f"GPU memory after model init: {torch.cuda.memory_allocated(self.gpu_id) / 1024**2:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to initialize models on {self.device}: {e}")
            
            # If GPU initialization fails, try on CPU
            if self.device.startswith("cuda"):
                logger.info("Falling back to CPU")
                self.device = "cpu"
                self.model = CosyVoice2(
                    'pretrained_models/CosyVoice2-0.5B',
                    load_jit=False,  # Disable JIT for CPU
                    load_trt=False,  # Disable TensorRT for CPU
                    fp16=False       # Use FP32 precision for CPU
                )
            else:
                raise  # Re-raise if already on CPU

    @staticmethod
    def _download_models():
        """
        Download pre-trained CosyVoice2 model files from ModelScope.
        
        Uses ModelScope's snapshot_download functionality to retrieve the
        pre-trained model files if they are not already present locally.
        """
        snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')

    def _get_prompt_path(self, voice_type: Optional[str] = None) -> Tuple[str, torch.Tensor]:
        """
        Get the path and processed audio tensor for a voice prompt.
        
        This method:
        - Resolves the voice type to a prompt file path
        - Checks the cache for previously processed prompts
        - Processes and caches new prompts as needed
        
        Args:
            voice_type: Type of voice to use (optional, uses default if not provided)
            
        Returns:
            Tuple of (file path, processed audio tensor)
            
        Raises:
            Exception: If the voice prompt file is not found or processing fails
        """
        # Resolve voice type, use default if not specified
        if not voice_type:
            voice_type = self.default_voice
            
        # Validate voice type
        if voice_type not in self.voice_prompts:
            logger.warning(f"Unknown voice type: {voice_type}, using default voice")
            voice_type = self.default_voice
            
        # Get path to voice prompt file
        voice_path = Path(project_dir) / self.voice_prompts[voice_type]
        if not voice_path.exists():
            raise Exception(f"Voice prompt file not found: {voice_path}")

        path = str(voice_path.resolve())

        # Check cache first for better performance
        if path in self._reference_audio_cache:
            return path, self._reference_audio_cache[path]

        # Process and cache if not found
        ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
        try:
            # Process the reference audio to the required format
            self.audio_processor.process_reference_audio(path, ref_audio)
            # Load the processed audio
            prompt_speech_16k = load_wav(ref_audio, 16000)
            # Clean up temporary file
            Path(ref_audio).unlink()

            # Cache the processed audio for future use
            self._reference_audio_cache[path] = prompt_speech_16k
            return path, prompt_speech_16k
        except Exception as e:
            logger.error(f"Reference audio processing failed: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(ref_audio):
                Path(ref_audio).unlink()
            raise

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize request parameters.
        
        Performs basic validation of request parameters and applies
        default values where appropriate.
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Check for required text parameter
        if not params.get('text'):
            raise ValueError("Missing required parameter: text")
        
        # Convert speed parameter to float with default value of 1.0
        params['speed'] = float(params.get('speed', 1.0))
        
        return params

    def _process_reference_audio(self, reference_path: str) -> torch.Tensor:
        """
        Process and cache user reference audio.
        
        Similar to _get_prompt_path but specifically for user-provided
        reference audio files used in voice cloning.
        
        Args:
            reference_path: Path to the reference audio file
            
        Returns:
            Processed audio tensor
            
        Raises:
            Exception: If audio processing fails
        """
        ref_path = str(Path(reference_path).resolve())
        
        # Check cache first for better performance
        if ref_path in self._reference_audio_cache:
            return self._reference_audio_cache[ref_path]
            
        # Process and cache if not found
        ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
        try:
            self.audio_processor.process_reference_audio(reference_path, ref_audio)
            prompt_speech_16k = load_wav(ref_audio, 16000)
            Path(ref_audio).unlink()
            
            # Cache the processed audio for future use
            self._reference_audio_cache[ref_path] = prompt_speech_16k
            return prompt_speech_16k
        except Exception as e:
            logger.error(f"Reference audio processing failed: {e}")
            if os.path.exists(ref_audio):
                Path(ref_audio).unlink()
            raise

    async def _create_streaming_response(self, text: str, prompt_speech_16k: torch.Tensor, speed: float) -> StreamingResponse:
        """
        Create a streaming audio response.
        
        Implements true streaming audio generation with on-the-fly MP3 encoding
        using FFmpeg. This allows for low-latency streaming of audio as it's
        being generated.
        
        Args:
            text: Text to synthesize
            prompt_speech_16k: Processed voice prompt tensor
            speed: Speech speed factor
            
        Returns:
            Streaming response with MP3 audio
        """
        async def audio_stream():
            """
            Generator function for streaming MP3 audio chunks.
            
            This asynchronous generator:
            1. Creates an FFmpeg process for PCM to MP3 conversion
            2. Sets up an async queue for MP3 data
            3. Generates audio chunks from the TTS model
            4. Feeds the audio to FFmpeg for conversion
            5. Yields the converted MP3 chunks as they become available
            """
            try:
                logger.info(f"Starting streaming audio generation for text: {text[:50]}...")
                
                # Create an FFmpeg process for PCM to MP3 conversion
                ffmpeg_process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-f", "s16le",          # Input format: signed 16-bit little-endian
                        "-ar", str(SAMPLE_RATE),  # Input sample rate
                        "-ac", "1",             # Input channels: mono
                        "-i", "pipe:0",         # Read from stdin
                        "-f", "mp3",            # Output format: MP3
                        "-codec:a", "libmp3lame",  # Use LAME encoder
                        "-b:a", "192k",         # Bitrate: 192kbps
                        "-write_xing", "0",     # Disable Xing header for better streaming
                        "pipe:1"                # Write to stdout
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0  # Unbuffered
                )
                
                # Create a queue for MP3 data
                mp3_queue = asyncio.Queue()
                
                # Create a task to read MP3 data from FFmpeg's stdout
                async def read_mp3_data():
                    """Reads MP3 data from FFmpeg and puts it in the queue."""
                    while True:
                        mp3_chunk = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: ffmpeg_process.stdout.read(4096)
                        )
                        if not mp3_chunk:
                            break
                        await mp3_queue.put(mp3_chunk)
                    await mp3_queue.put(None)  # Signal end of data
                
                # Start the MP3 reader task
                mp3_reader_task = asyncio.create_task(read_mp3_data())
                
                try:
                    # Process audio chunks and feed them to FFmpeg
                    for _, j in enumerate(
                        self.model.inference_cross_lingual(
                            text, prompt_speech_16k, stream=True, speed=speed
                        )
                    ):
                        if 'tts_speech' in j and j['tts_speech'] is not None:
                            audio = j['tts_speech']
                            if audio is not None and audio.numel() > 0:
                                # Process audio on GPU as much as possible
                                audio = self.audio_processor.normalize_audio(audio)
                                # Convert float32 [-1, 1] to int16 PCM format
                                audio_int16 = (audio * 32767).to(torch.int16)
                                # Move to CPU only when needed
                                audio_int16_cpu = audio_int16.cpu()
                                # Convert to bytes
                                audio_bytes = audio_int16_cpu.numpy().tobytes()
                                
                                # Write to FFmpeg's stdin
                                await asyncio.get_event_loop().run_in_executor(
                                    None, lambda: ffmpeg_process.stdin.write(audio_bytes)
                                )
                                
                                # Yield any available MP3 data
                                while not mp3_queue.empty():
                                    mp3_chunk = await mp3_queue.get()
                                    if mp3_chunk is None:
                                        break
                                    yield mp3_chunk
                    
                    # Close FFmpeg's stdin to signal end of input
                    ffmpeg_process.stdin.close()
                    
                    # Yield any remaining MP3 data
                    while True:
                        mp3_chunk = await mp3_queue.get()
                        if mp3_chunk is None:
                            break
                        yield mp3_chunk
                    
                    # Wait for the MP3 reader task to complete
                    await mp3_reader_task
                    
                    logger.info("Audio streaming completed successfully")
                finally:
                    # Clean up FFmpeg process
                    try:
                        if ffmpeg_process.poll() is None:
                            ffmpeg_process.terminate()
                            ffmpeg_process.wait(timeout=5)
                    except Exception as e:
                        logger.error(f"Error terminating FFmpeg process: {e}")
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                raise

        # Return a proper MP3 streaming response with appropriate headers
        return StreamingResponse(
            audio_stream(),
            media_type='audio/mpeg',
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Disposition": "attachment; filename=\"generated_audio.mp3\"",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache, no-store",
            }
        )

    async def _handle_tts(self, params: Dict[str, Any], outname: str) -> str:
        """
        Handle standard TTS request.
        
        Generates speech using a specified or default voice prompt.
        
        Args:
            params: Request parameters including text and voice settings
            outname: Output filename
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If audio generation fails
        """
        # Get voice prompt
        voice_type = params.get('voice_type')
        _, prompt_speech_16k = self._get_prompt_path(voice_type)
        
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
                if audio is not None and audio.numel() > 0:
                    audio = self.audio_processor.normalize_audio(audio)
                    audio = audio.cpu()
                    audio_list.append(audio)

        if not audio_list:
            raise Exception("Failed to generate valid audio")

        # Concatenate audio segments
        audio_data = torch.cat(audio_list, dim=1)
        
        # Save to output file
        output_path = f"{tmp_dir}/{outname}"
        return await AudioProcessor.save_audio(audio_data, output_path)

    async def _handle_zero_shot(self, params: Dict[str, Any], outname: str) -> str:
        """
        Handle zero-shot voice cloning request.
        
        Clones a voice from a reference audio file and uses it to synthesize speech.
        
        Args:
            params: Request parameters including text, reference audio, and reference text
            outname: Output filename
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If required parameters are missing or audio generation fails
        """
        # Validate required parameters
        if not params.get('reference_text'):
            raise Exception("Missing required parameter: reference_text")
        if not params.get('reference_audio') or not os.path.exists(params['reference_audio']):
            raise Exception(f"Reference audio not found: {params['reference_audio']}")
            
        # Process reference audio
        prompt_speech_16k = self._process_reference_audio(params['reference_audio'])
        
        # Generate audio
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
                    audio_list.append(audio)
                else:
                    logger.warning("Skipping invalid audio segment")

        if not audio_list:
            raise Exception("Failed to generate valid audio")

        # Concatenate audio segments
        audio_data = torch.cat(audio_list, dim=1)
        
        # Save to output file
        output_path = f"{tmp_dir}/{outname}"
        return await AudioProcessor.save_audio(audio_data, output_path)

    async def _handle_cross_lingual(self, params: Dict[str, Any], outname: str) -> str:
        """
        Handle cross-lingual voice cloning request.
        
        Clones a voice from a reference audio file and uses it to synthesize speech
        in any supported language.
        
        Args:
            params: Request parameters including text and reference audio
            outname: Output filename
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If reference audio is missing or audio generation fails
        """
        # Validate required parameters
        if not params.get('reference_audio') or not os.path.exists(params['reference_audio']):
            raise Exception(f"Reference audio not found: {params['reference_audio']}")
            
        # Process reference audio
        prompt_speech_16k = self._process_reference_audio(params['reference_audio'])
        
        # Generate audio
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
        
        # Save to output file
        output_path = f"{tmp_dir}/{outname}"
        return await AudioProcessor.save_audio(audio_data, output_path)

    async def _handle_instruct(self, params: Dict[str, Any], outname: str) -> str:
        """
        Handle instruction-based TTS request.
        
        Generates speech with specific voice characteristics based on natural
        language instructions.
        
        Args:
            params: Request parameters including text, instruction, and voice settings
            outname: Output filename
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If instruction is missing or audio generation fails
        """
        # Validate required parameters
        if not params.get('instruction'):
            raise Exception("Missing required parameter: instruction")

        # Get reference audio
        voice_type = params.get('voice_type')
        _, prompt_speech_16k = self._get_prompt_path(voice_type)

        # Generate audio
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
        
        # Save to output file
        output_path = f"{tmp_dir}/{outname}"
        return await AudioProcessor.save_audio(audio_data, output_path)

    async def batch(self, tts_type: str, outname: str, params: Dict[str, Any]) -> str:
        """
        Process a batch TTS request based on the type.
        
        This method dispatches requests to the appropriate handler based on
        the TTS type.
        
        Args:
            tts_type: Type of TTS operation ('tts', 'zero_shot', 'cross_lingual', 'instruct')
            outname: Output filename
            params: Request parameters
            
        Returns:
            Path to the generated audio file
            
        Raises:
            ValueError: If an unsupported TTS type is specified
        """
        if tts_type == 'tts':
            return await self._handle_tts(params, outname)
        elif tts_type == 'zero_shot':
            return await self._handle_zero_shot(params, outname)
        elif tts_type == 'cross_lingual':
            return await self._handle_cross_lingual(params, outname)
        elif tts_type == 'instruct':
            return await self._handle_instruct(params, outname)
        else:
            raise ValueError(f"Unsupported TTS type: {tts_type}")

    async def __call__(self, request: Request) -> Response:
        """
        Main request handler for the Ray Serve deployment.
        
        This method:
        - Handles HTTP requests to the API endpoints
        - Processes different TTS operations based on the endpoint
        - Supports both streaming and non-streaming responses
        - Provides CORS support for cross-origin requests
        - Implements comprehensive error handling
        
        Args:
            request: The incoming HTTP request
            
        Returns:
            HTTP response with generated audio or error information
        """
        # Handle CORS preflight requests
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
                # Parse request parameters from JSON or form data
                if request.headers.get("content-type") == "application/json":
                    params = await request.json()
                else:
                    form = await request.form()
                    params = dict(form)

                # Extract endpoint from URL path
                path = request.url.path.strip('/')
                parts = path.split('/')
                endpoint = parts[-1] if parts else 'tts'  # Default to 'tts' if no endpoint

                logger.info(f"Processing request: {endpoint}, parameters: {params}")

                # Handle health check endpoint
                if endpoint == "healthcheck":
                    return JSONResponse({"status": "healthy"})

                # Validate endpoint
                if endpoint not in ["tts", "zero_shot", "cross_lingual", "instruct"]:
                    return JSONResponse(
                        {"error": "Invalid endpoint"},
                        status_code=404
                    )

                # Validate request parameters
                params = self._validate_params(params)

                # Handle streaming if requested
                if params.get('stream', False):
                    voice_type = params.get('voice_type')
                    _, prompt_speech_16k = self._get_prompt_path(voice_type)
                    return await self._create_streaming_response(params['text'], prompt_speech_16k, params['speed'])
                else:
                    # Generate audio based on endpoint
                    outname = f"{endpoint}-{int(time.time())}.mp3"
                    output_path = await self.batch(endpoint, outname, params)

                    # Read the generated audio file
                    with open(output_path, 'rb') as f:
                        audio_content = f.read()

                    # Clean up temporary file
                    try:
                        Path(output_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete output file {output_path}: {e}")

                    # Return audio response
                    return Response(
                        audio_content,
                        media_type='audio/mpeg',
                        headers=headers
                    )

        except Exception as e:
            # Log and return error response
            logger.error(f"Request processing failed: {e}")
            return JSONResponse(
                {"error": str(e)},
                status_code=400,
                headers=headers
            )


def main():
    """
    Start the CosyVoice Ray Serve API service.
    
    This function:
    - Detects available GPUs
    - Connects to or initializes a Ray cluster
    - Starts the Ray Serve service
    - Deploys the CosyVoiceService with appropriate resources
    - Configures routing and service options
    """
    try:
        # Get GPU count for resource allocation
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} available GPU(s)")

        # If no GPUs are available, use CPU mode
        if gpu_count == 0:
            logger.warning("No GPUs available, will use CPU")
            gpu_count = 1  # Use one replica on CPU

        # Connect to existing Ray cluster or initialize a new one
        if not ray.is_initialized():
            ray.init(
                address="auto",  # Connect to existing cluster or start a new one
                namespace="serve",  # Use 'serve' namespace
                ignore_reinit_error=True,  # Ignore errors if already initialized
                runtime_env={
                    "env_vars": {
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_LAUNCH_BLOCKING": "1"
                    }
                }
            )
            logger.info("Connected to Ray cluster")

        # Start Ray Serve
        serve.start(
            detached=True,  # Run in detached mode
            http_options={
                "host": "0.0.0.0",  # Listen on all interfaces
                "port": 9998,       # HTTP port
                "location": "HeadOnly"  # Run HTTP server on head node only
            }
        )

        # Deploy CosyVoice service
        logger.info(f"Deploying CosyVoice Service with {gpu_count} replica(s)")

        # Create deployment with appropriate resources
        deployment = CosyVoiceService.options(
            num_replicas=gpu_count,  # One replica per GPU, or one for CPU mode
            ray_actor_options={
                "num_cpus": 1,  # Allocate one CPU per replica
                "num_gpus": 1 if gpu_count > 0 and torch.cuda.is_available() else 0,  # Allocate one GPU per replica if available
                "memory": 4 * 1024 * 1024 * 1024,  # Allocate 4GB memory per replica
            }
        ).bind()

        # Run service with routing configuration
        serve.run(
            deployment,
            route_prefix="/v1/model/cosyvoice",  # API route prefix
            name="cosyvoice_service"             # Service name
        )

        logger.info("Service running. API available at: /v1/model/cosyvoice")

    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
