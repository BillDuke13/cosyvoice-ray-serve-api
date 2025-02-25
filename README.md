# CosyVoice Ray Serve API

This project provides a Ray Serve-based HTTP API wrapper around [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), a high-quality Text-to-Speech (TTS) system. It extends the original project with features like RESTful API, Docker deployment, health monitoring, and streaming audio generation.

## Overview

CosyVoice Ray Serve API enhances the original CosyVoice project by providing a scalable, production-ready API service for text-to-speech synthesis. The implementation uses Ray Serve for deployment management and scaling, with automatic GPU detection and allocation. It includes comprehensive error handling, logging, and resource management.

Key enhancements include:
- RESTful API interface using Ray Serve
- Containerized deployment with Docker
- Health monitoring and graceful shutdown
- Efficient GPU resource management
- Streaming audio generation
- Reference audio caching for improved performance

## Features

All core TTS capabilities are provided by the original CosyVoice project:

- **Zero-shot Voice Cloning**: Clone any voice with a single audio sample
- **Cross-lingual Synthesis**: Maintain voice identity across different languages
- **Instruction-based Synthesis**: Control speech characteristics with natural language
- **Fine-grained Control**: Adjust emotions, prosody, and speaking style
- **Multiple Language Support**:
  - Chinese
  - English
  - Japanese
  - Cantonese
  - Korean
- **Multiple Voice Types**: Support for various voice types (male/female)
- **Adjustable Speech Speed**: Control the pace of speech generation
- **Streaming Audio Generation**: Real-time audio synthesis for low-latency applications

## Getting Started

### Prerequisites

- Python 3.11 (recommended)
- FFmpeg
- Sox and Sox development packages
- CUDA 11.7+ (optional, for GPU acceleration)
- Ray
- PyTorch 2.0+
- Torchaudio

### Installation

#### Option 1: Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/BillDuke13/cosyvoice-ray-serve-api.git
   cd cosyvoice-ray-serve-api
   ```

2. Create and activate conda environment:

   ```bash
   conda create -n cosyvoice-ray-serve-api -y python=3.11
   conda activate cosyvoice-ray-serve-api
   ```

3. Install system dependencies:

   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y ffmpeg sox libsox-fmt-all libsox-dev

   # CentOS/RHEL
   sudo yum install -y epel-release
   sudo yum install -y ffmpeg sox sox-devel
   ```

4. Install Python dependencies:

   ```bash
   conda install -y -c conda-forge pynini==2.1.5
   pip install -r requirements.txt
   ```
5. Download models (automatic on first run)

#### Option 2: Docker Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/BillDuke13/cosyvoice-ray-serve-api.git
   cd cosyvoice-ray-serve-api
   ```

2. Build the Docker image:

   ```bash
   docker build -t cosyvoice-ray-serve-api .
   ```

3. Run the container:

   ```bash
   docker run -d --name cosyvoice-api \
     --gpus all \
     -p 9998:9998 \
     -p 8265:8265 \
     -v $(pwd)/asset:/app/asset \
     -v $(pwd)/logs:/app/logs \
     cosyvoice-ray-serve-api
   ```

   This will:

   - Mount the local `asset` directory to provide voice prompts.
   - Mount the local `logs` directory to persist logs.
   - Expose the API on port 9998.
   - Expose the Ray dashboard on port 8265.

## Usage

### Starting the Server

1. Start the Ray cluster:

   ```bash
   ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379 --temp-dir=/tmp/ray
   ```

2. Set the Ray address and start the service:

   ```bash
   python cosyvoice/api.py
   ```

The server will be accessible at `http://localhost:9998`.

### API Endpoints

The service exposes several HTTP endpoints for different TTS operations. All endpoints support both regular and streaming responses.

#### 1. Standard TTS (`/v1/model/cosyvoice/tts`)

Synthesizes speech from text using a specified voice type.

**Method:** `POST`

**Request Body:**

```json
{
    "text": "Text to synthesize", // Required
    "voice_type": "qwen",        // Optional (default: "qwen")
    "speed": 1.0,                // Optional (default: 1.0)
    "stream": false              // Optional, enable streaming output (default: false)
}
```

**Response:**

For non-streaming requests:

- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

For streaming requests:

- Content-Type: `audio/wav`
- Body: Chunked audio stream

#### 2. Zero-shot Voice Cloning (`/v1/model/cosyvoice/zero_shot`)

Clones a voice from a reference audio file and uses it to speak text.

**Method:** `POST`

**Request Body:**

```json
{
    "text": "Text to synthesize",          // Required
    "reference_audio": "path/to/audio.wav", // Required
    "reference_text": "Reference text",    // Required
    "speed": 1.0,                          // Optional (default: 1.0)
    "stream": false                        // Optional, enable streaming output (default: false)
}
```

**Response:**

- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

#### 3. Cross-lingual Voice Cloning (`/v1/model/cosyvoice/cross_lingual`)

Clones a voice from a reference audio file and uses it to speak text in any supported language.

**Method:** `POST`

**Request Body:**

```json
{
    "text": "Text to synthesize",          // Required
    "reference_audio": "path/to/audio.wav", // Required
    "speed": 1.0,                          // Optional (default: 1.0)
    "stream": false                        // Optional, enable streaming output (default: false)
}
```

**Response:**

- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

#### 4. Instruction-based TTS (`/v1/model/cosyvoice/instruct`)

Synthesizes speech from text with specific instructions on voice style.

**Method:** `POST`

**Request Body:**

```json
{
    "text": "Text to synthesize",  // Required
    "instruction": "Instruction",  // Required
    "voice_type": "qwen",          // Optional (default: "qwen")
    "speed": 1.0                   // Optional (default: 1.0)
}
```

**Response:**

- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

#### 5. Health Check (`/v1/model/cosyvoice/healthcheck`)

Checks if the service is running properly.

**Method:** `POST`

**Response:**
- Content-Type: `application/json`
- Body: `{"status": "healthy"}`

## Architecture

### Ray Serve Integration

The service is implemented as a Ray Serve deployment class (`CosyVoiceService`) that:

1. Initializes the CosyVoice model.
2. Handles HTTP requests.
3. Processes audio data.
4. Manages GPU resources.
5. Provides streaming and non-streaming responses.

### GPU Resource Management

The service includes GPU resource management:

- Automatic GPU detection.
- GPU assignment to service replicas.
- Optimized CUDA memory allocation.
- Fallback to CPU if GPU initialization fails.
- Proper resource cleanup on shutdown.

### Audio Processing Pipeline

The audio processing pipeline includes:

1. Text preprocessing.
2. Reference audio processing (for voice cloning).
3. Voice synthesis using the CosyVoice model.
4. Post-processing (normalization, validation).
5. Format conversion to WAV.
6. Streaming support.

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU devices to use (e.g., "0,1" or "" for CPU-only).
- `CUDA_DEVICE_ORDER`: Device ordering (default: "PCI_BUS_ID").
- `CUDA_LAUNCH_BLOCKING`: CUDA error debugging (default: "1").
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch CUDA memory allocation settings (default: "max_split_size_mb:512").
- `RAY_ADDRESS`: Ray cluster address (default: "ray://localhost:6379").
- `RAY_TEMP_DIR`: Ray temporary directory (default: "/tmp/ray").

### Project Structure

```
cosyvoice-ray-serve-api/
├── cosyvoice/
│   ├── api.py          # Main API implementation
│   ├── webui.py        # Web interface (optional)
│   └── cosyvoice/      # Core CosyVoice implementation
├── asset/              # Voice prompt files
│   └── qwen.wav        # Default voice prompt
├── logs/               # Log files
├── tmp/                # Temporary files
├── pretrained_models/  # Downloaded model files
├── docs/               # Documentation
│   ├── API.md          # API documentation
│   └── DOCKER.md       # Docker implementation details
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues:**
   - **Symptom:** Out of memory errors or CUDA errors.
   - **Solution:** Adjust `PYTORCH_CUDA_ALLOC_CONF`, reduce batch size, or use CPU mode.
2. **Ray Connection Issues:**
   - **Symptom:** Cannot connect to Ray cluster or service fails to start.
   - **Solution:** Ensure Ray is running (`ray status`), check `RAY_ADDRESS`, restart Ray cluster.
3. **Audio Processing Errors:**
   - **Symptom:** FFmpeg or Sox related errors.
   - **Solution:** Verify system dependencies, check input audio format, ensure sufficient disk space.
4. **Model Loading Errors:**
   - **Symptom:** Model fails to load or initialize.
   - **Solution:** Check internet connection, verify disk space, try CPU-only mode.

### Logs

Check the following log files for troubleshooting:

- Application logs: `logs/cosyvoice.log`
- Ray dashboard: `http://localhost:8265`
- Docker logs: `docker logs cosyvoice-api`

## Documentation

- [API Documentation](docs/API.md): Detailed technical documentation of the API implementation.
- [Docker Documentation](docs/DOCKER.md): Information about the Docker implementation and deployment.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure they are well-documented and tested.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the core TTS models.
- [Ray Project](https://ray.io/) for the serving framework.
- [ModelScope](https://modelscope.cn/) for model hosting.
- FFmpeg and Sox for audio processing.
