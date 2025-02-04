"""CosyVoice Text-to-Speech Service API.

A Ray Serve based HTTP API for text-to-speech synthesis using CosyVoice models.
Supports standard TTS, voice cloning, and cross-lingual synthesis.

Usage:
    $ python api.py

The server provides the following endpoints:
    - /v1/model/cosyvoice/tts: Standard text-to-speech synthesis
    - /v1/model/cosyvoice/clone: Cross-lingual voice cloning
    - /v1/model/cosyvoice/clone_eq: Same-language voice cloning

Example:
    $ curl -X POST http://localhost:9233/v1/model/cosyvoice/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "speaker": "default", "language": "en"}' \
        --output output.wav
"""

import os
import sys
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
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from modelscope import snapshot_download
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Constants for voice types and role mapping
VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
DEFAULT_ROLE_MAP = {
    ('default', 'zh'): '中文女',  # Chinese female
    ('default', 'en'): '英文女',  # English female
    ('default', 'jp'): '日语男'   # Japanese male
}

class AudioProcessor:
    """Audio file processing utility class.
    
    Provides static methods for processing audio files, including format conversion
    and sample rate adjustment.
    """

    @staticmethod
    def process_reference_audio(audio_path: str, output_path: str, sample_rate: int = 16000) -> None:
        """Processes a reference audio file to match required format.

        Args:
            audio_path: Path to the source audio file.
            output_path: Path where processed audio will be saved.
            sample_rate: Target sample rate in Hz. Defaults to 16000.

        Raises:
            subprocess.CalledProcessError: If FFmpeg processing fails.
        """
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", 
             "-i", audio_path, "-ar", str(sample_rate), output_path],
            check=True, capture_output=True, text=True
        )

# Set up paths
root_dir = Path(__file__).parent.as_posix()
tmp_dir = Path(f'{root_dir}/tmp').as_posix()
logs_dir = Path(f'{root_dir}/logs').as_posix()
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

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

# Configure environment
if sys.platform == 'win32':
    os.environ['PATH'] = f"{root_dir};{root_dir}\\ffmpeg;{root_dir}/third_party/Matcha-TTS;" + os.environ['PATH']
else:
    os.environ['PATH'] = f"{root_dir}:{root_dir}/ffmpeg:" + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')

@serve.deployment(name="tts_service")
class TTSDeployment:
    """Ray Serve deployment for text-to-speech service.
    
    Handles all TTS operations including standard synthesis, voice cloning,
    and cross-lingual synthesis. Uses CosyVoice models for high-quality
    speech synthesis.

    Attributes:
        sft_model: CosyVoice model for standard TTS operations.
        tts_model: CosyVoice2 model for voice cloning operations.
        audio_processor: Utility instance for audio processing.
    """

    def __init__(self) -> None:
        """Initialize models and resources."""
        self._setup_models()
        self.audio_processor = AudioProcessor()

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
                endpoint = path.split('/')[-1] if path else 'tts'  # Default to tts if no path
                
                logger.info(f"Processing request for endpoint: {endpoint} with params: {params}")
                
                if path == "v1/model/cosyvoice/tts" or not path:  # Handle root path as TTS
                    outname = f"tts-{int(time.time())}.wav"
                    output_path = await self.batch('tts', outname, params)
                elif path in ["v1/model/cosyvoice/clone", "v1/model/cosyvoice/clone_mul"]:
                    outname = f"clone-{int(time.time())}.wav"
                    output_path = await self.batch('clone', outname, params)
                elif path == "v1/model/cosyvoice/clone_eq":
                    outname = f"clone-eq-{int(time.time())}.wav"
                    output_path = await self.batch('clone_eq', outname, params)
                else:
                    return JSONResponse({"error": "Invalid endpoint"}, status_code=404)

                return Response(
                    open(output_path, 'rb').read(),
                    media_type='audio/x-wav'
                )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(
                {"error": str(e), "message": "Please provide required parameters: text, and optionally speaker/language or role"},
                status_code=400
            )

    def _setup_models(self) -> None:
        """Download and initialize TTS models."""
        # Download models
        self._download_models()
        
        # Initialize on appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.sft_model = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=True)
        self.tts_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False)

    @staticmethod
    def _download_models() -> None:
        """Download required model files."""
        snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
        snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')

    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validates and transforms input parameters.

        Ensures all required parameters are present and applies default values
        where necessary. Maps speaker and language preferences to appropriate
        voice roles.

        Args:
            params: Dictionary containing request parameters.

        Returns:
            Dictionary with validated and transformed parameters.

        Raises:
            ValueError: If required parameters are missing.
        """
        if not params.get('text'):
            raise ValueError("Missing required parameter: text")
            
        # Map speaker/language to role if not provided
        if not params.get('role'):
            speaker = params.get('speaker', 'default')
            language = params.get('language', 'zh')
            
            # Default mapping
            role_map = {
                ('default', 'zh'): '中文女',
                ('default', 'en'): '英文女',
                ('default', 'jp'): '日语男',
            }
            params['role'] = role_map.get((speaker, language), '中文女')
            
        # Add default speed if not provided
        params['speed'] = float(params.get('speed', 1.0))
        
        return params

    async def batch(self, tts_type: str, outname: str, params: Dict[str, Any]) -> str:
        """Processes a batch text-to-speech request.

        Handles different types of TTS operations including standard synthesis,
        voice cloning, and cross-lingual synthesis. Manages audio file processing
        and model inference.

        Args:
            tts_type: Type of TTS operation ('tts', 'clone', or 'clone_eq').
            outname: Output filename for the generated audio.
            params: Request parameters including text and voice settings.

        Returns:
            Path to the generated audio file.

        Raises:
            Exception: If audio processing or synthesis fails.
        """
        prompt_speech_16k = None
        if tts_type != 'tts':
            if not params['reference_audio'] or not os.path.exists(f"{root_dir}/{params['reference_audio']}"):
                raise Exception(f'Reference audio not found: {params["reference_audio"]}')
                
            ref_audio = f"{tmp_dir}/-refaudio-{time.time()}.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-ignore_unknown", "-y", "-i", params['reference_audio'], 
                     "-ar", "16000", ref_audio],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    check=True
                )
                prompt_speech_16k = load_wav(ref_audio, 16000)
            except Exception as e:
                raise Exception(f'Failed to process reference audio: {e}')

        text = params['text']
        audio_list = []
        
        if tts_type == 'tts':
            for _, j in enumerate(self.sft_model.inference_sft(text, params['role'], 
                                                             stream=False, speed=params['speed'])):
                audio_list.append(j['tts_speech'])
        elif tts_type == 'clone_eq' and params.get('reference_text'):
            for _, j in enumerate(self.tts_model.inference_zero_shot(text, params.get('reference_text'),
                                                                   prompt_speech_16k, stream=False, 
                                                                   speed=params['speed'])):
                audio_list.append(j['tts_speech'])
        else:
            for _, j in enumerate(self.tts_model.inference_cross_lingual(text, prompt_speech_16k,
                                                                       stream=False, speed=params['speed'])):
                audio_list.append(j['tts_speech'])

        audio_data = torch.concat(audio_list, dim=1)
        sample_rate = 22050 if tts_type == 'tts' else 24000
        output_path = f"{tmp_dir}/{outname}"
        torchaudio.save(output_path, audio_data, sample_rate, format="wav")
        
        return output_path

def setup_logging() -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{logs_dir}/{datetime.datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

def connect_or_start_ray() -> None:
    """Initializes Ray by connecting to existing cluster or starting locally.

    Attempts to connect to a remote Ray cluster if RAY_ADDRESS environment variable
    is set. Otherwise, tries to connect to an existing local instance or starts
    a new one.

    Raises:
        Exception: If Ray initialization fails.
    """
    ray_address = os.getenv('RAY_ADDRESS')
    try:
        if ray_address:
            # Connect to specified remote Ray cluster
            logger.info(f"Attempting to connect to Ray cluster at {ray_address}")
            ray.init(address=ray_address)
            logger.info("Successfully connected to remote Ray cluster")
        else:
            # Check for existing local Ray instance
            try:
                ray.init(address='auto')
                logger.info("Connected to existing local Ray instance")
            except ConnectionError:
                # Start new instance if none exists
                logger.info("Starting new local Ray instance")
                ray.init(num_gpus=1 if torch.cuda.is_available() else 0)
                logger.info("Local Ray instance started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise

def main() -> None:
    """Initializes and runs the TTS service.

    Sets up logging, initializes Ray, and starts the Ray Serve deployment.
    Handles graceful shutdown on keyboard interrupt.
    """
    setup_logging()
    
    # Connect to Ray cluster or start local instance
    connect_or_start_ray()
    
    try:
        serve.start(http_options={"host": "0.0.0.0", "port": 9233})
        serve.run(TTSDeployment.bind())
        
        print("Ray Serve TTS API is running on http://localhost:9233")
        
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error running service: {e}")
        raise
    except KeyboardInterrupt:
        print("Shutting down...")
        serve.shutdown()
        ray.shutdown()

if __name__ == "__main__":
    main()