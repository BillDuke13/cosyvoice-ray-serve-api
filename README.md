# CosyVoice Ray Serve API

This project provides a Ray Serve-based HTTP API wrapper around [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), a high-quality Text-to-Speech (TTS) system.

## Overview

This API service extends the original CosyVoice project by:
- Adding a RESTful API interface using Ray Serve
- Providing containerized deployment
- Adding health monitoring and graceful shutdown
- Implementing efficient resource management

## Features

All core TTS capabilities are provided by the original CosyVoice project:
- High-quality speech synthesis
- Cross-lingual voice cloning with accent preservation 
- Same-language voice cloning
- Support for multiple languages:
  - Chinese
  - English
  - Japanese
  - Cantonese
  - Korean
- Multiple voice types (male/female)
- Adjustable speech speed

## Table of Contents
- [CosyVoice Ray Serve API](#cosyvoice-ray-serve-api)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Starting the Server](#starting-the-server)
    - [API Endpoints](#api-endpoints)
      - [1. Standard TTS (`/v1/model/cosyvoice/tts`)](#1-standard-tts-v1modelcosyvoicetts)
      - [2. Zero-shot Voice Cloning (`/v1/model/cosyvoice/zero_shot`)](#2-zero-shot-voice-cloning-v1modelcosyvoicezero_shot)
      - [3. Cross-lingual Voice Cloning (`/v1/model/cosyvoice/cross_lingual`)](#3-cross-lingual-voice-cloning-v1modelcosyvoicecross_lingual)
      - [4. Instruction-based TTS (`/v1/model/cosyvoice/instruct`)](#4-instruction-based-tts-v1modelcosyvoiceinstruct)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Project Structure](#project-structure)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Logs](#logs)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

- Python 3.11 (recommended)
- FFmpeg
- Sox and Sox development packages
- CUDA 11.7+ (optional, for GPU acceleration)
- Ray
- PyTorch 2.0+
- Torchaudio

## Installation

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
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

5. Download models (automatic on first run)

## Usage

### Starting the Server

1. Start the Ray cluster:
```bash
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379 --temp-dir=/data/ray
```

2. Set the Ray address and start the service:
```bash
python cosyvoice/api.py
```

The server will be accessible at `http://localhost:9998`.

### API Endpoints

The service exposes several HTTP endpoints for different TTS operations. All endpoints support both regular and streaming responses.

#### 1. Standard TTS (`/v1/model/cosyvoice/tts`)

Synthesizes speech from text using a specified voice type. This is the most basic and efficient endpoint for standard TTS operations.

**Features:**
- Streaming support for real-time audio generation
- Voice type selection with default "qwen" voice
- Adjustable speech speed
- Automatic audio normalization and validation
- Reference audio caching for improved performance

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",        // Required: Text to convert to speech
    "voice_type": "qwen",               // Optional: Voice type to use (default: "qwen")
    "speed": 1.0,                       // Optional: Speech speed multiplier (default: 1.0)
    "stream": false                     // Optional: Enable streaming output (default: false)
}
```

**Response:**
For non-streaming requests:
- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate, 16-bit PCM)

For streaming requests:
- Content-Type: `audio/wav`
- Body: Chunked audio stream

**Error Responses:**
- 400: Invalid request parameters or missing required parameter
- 404: Voice type not found or invalid endpoint
- 500: Internal server error (audio processing, model inference, etc.)

**Example:**
```bash
curl -X POST http://localhost:9998/v1/model/cosyvoice/tts \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello world",
        "voice_type": "qwen",
        "speed": 1.0
    }' \
    --output output.wav
```

**Response:** The API returns a WAV file containing the synthesized speech.

#### 2. Zero-shot Voice Cloning (`/v1/model/cosyvoice/zero_shot`)
Clones a voice from a reference audio file and uses it to speak text, given the reference audio and its corresponding text. This endpoint enables voice cloning without requiring extensive training data.

**Features:**
- High-quality voice cloning from a single audio sample
- Preserves speaker characteristics and style
- Automatic reference audio processing and caching
- Supports adjustable speech speed
- Robust audio validation and error handling

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",             // Required: Text to speak
    "reference_audio": "path/to/audio.wav",   // Required: Reference voice audio file
    "reference_text": "Reference text",       // Required: Text content of reference audio
    "speed": 1.0,                            // Optional: Speech speed multiplier (default: 1.0)
    "stream": false                          // Optional: Enable streaming output (default: false)
}
```

**Response:**
For non-streaming requests:
- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate, 16-bit PCM)

For streaming requests:
- Content-Type: `audio/wav`
- Body: Chunked audio stream

**Technical Details:**
- Reference audio is automatically processed to 16kHz mono format
- Audio is validated for quality and format compliance
- Processed reference audio is cached for improved performance
- Supports various input audio formats through FFmpeg conversion

**Error Responses:**
- 400: Missing required parameters or invalid audio file
- 404: Reference audio file not found
- 500: Audio processing or synthesis error (with detailed error messages)

**Example:**
```bash
curl -X POST http://localhost:9998/v1/model/cosyvoice/zero_shot \
    -H "Content-Type: application/json" \
    -d '{
        "text": "你好，世界",
        "reference_audio": "path/to/reference.wav",
        "reference_text": "Reference text",
        "speed": 1.0
    }' \
    --output output.wav
```

#### 3. Cross-lingual Voice Cloning (`/v1/model/cosyvoice/cross_lingual`)

Clones a voice from a reference audio file and uses it to speak text in any supported language.

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",             // Required: Text to speak in target language
    "reference_audio": "path/to/audio.wav",   // Required: Reference voice audio file
    "speed": 1.0                             // Optional: Speech speed multiplier (default: 1.0)
}
```

**Response:**
- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

**Error Responses:**
- 400: Missing required parameters or invalid audio file
- 500: Audio processing or synthesis error

**Example:**
```bash
curl -X POST http://localhost:9998/v1/model/cosyvoice/cross_lingual \
    -H "Content-Type: application/json" \
    -d '{
        "text": "你好，世界",
        "reference_audio": "path/to/reference.wav",
        "speed": 1.0
    }' \
    --output output.wav
```

**Response:** The API returns a WAV file containing the synthesized speech with the cloned voice.

#### 4. Instruction-based TTS (`/v1/model/cosyvoice/instruct`)
Synthesizes speech from text with specific instructions on voice style.

**Method:** `POST`

**Request Body:**
```json
{
"text": "Text to synthesize", // Required: Text to speak
"instruction": "Instruction", // Required: Instruction for voice style
"voice_type": "qwen", // Optional: Voice type (default: "qwen")
"speed": 1.0 // Optional: Speech speed multiplier (default: 1.0)
}
```

**Response:**
- Content-Type: `audio/x-wav`
- Body: WAV audio file (24kHz sample rate)

**Error Responses:**
- 400: Missing required parameters
- 500: Audio processing or synthesis error

**Example:**
```bash
curl -X POST http://localhost:9998/v1/model/cosyvoice/instruct \
-H "Content-Type: application/json" \
-d '{
"text": "Hello world",
"instruction": "Speak with a happy tone",
"voice_type": "qwen",
"speed": 1.0
}' \
--output output.wav
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU devices to use (e.g., "0,1" or "" for CPU-only)
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch CUDA memory allocation settings

### Project Structure

```
cosyvoice-ray-serve-api/
├── cosyvoice/
│   ├── api.py          # Main API implementation
│   └── webui.py        # Web interface (optional)
├── asset/              # Voice prompt files
├── logs/               # Log files
├── tmp/                # Temporary files
└── pretrained_models/  # Downloaded model files
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Symptom: Out of memory errors
   - Solution: Adjust batch size or use CPU mode

2. **Ray Connection Issues**
   - Symptom: Cannot connect to Ray cluster
   - Solution: Ensure Ray is running and RAY_ADDRESS is correct

3. **Audio Processing Errors**
   - Symptom: FFmpeg or Sox related errors
   - Solution: Verify system dependencies are installed

### Logs

Check the following log files for troubleshooting:
- Application logs: `logs/YYYYMMDD.log`
- Ray dashboard: `http://localhost:8265`

## Contributing

We welcome contributions to CosyVoice! If you'd like to contribute, please follow these guidelines:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and ensure they are well-documented and tested
4. Submit a pull request with a clear description of your changes

## License

This project is licensed under the Apache 2.0 License. For more details, please refer to the [LICENSE](LICENSE) file.

## Acknowledgments

- CosyVoice models from ModelScope
- Ray Project for the serving framework
- FFmpeg and Sox for audio processing
