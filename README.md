# CosyVoice Ray Serve API

A high-performance Text-to-Speech (TTS) service built with Ray Serve, supporting multiple synthesis modes including standard TTS, voice cloning, and cross-lingual synthesis.

## Features

- Standard text-to-speech synthesis
- Cross-lingual voice cloning
- Same-language voice cloning
- Support for multiple languages (Chinese, English, Japanese, Korean)
- Multiple voice types (male/female)
- Adjustable speech speed
- RESTful API interface

## Prerequisites

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

```bash
python cosyvoice/api.py
```

The server will start on `http://localhost:9233`

### API Endpoints

#### 1. Standard TTS
```bash
curl -X POST http://localhost:9233/v1/model/cosyvoice/tts \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello world",
        "speaker": "default",
        "language": "en",
        "speed": 1.0
    }' \
    --output output.wav
```

#### 2. Cross-lingual Voice Cloning
```bash
curl -X POST http://localhost:9233/v1/model/cosyvoice/clone \
    -H "Content-Type: application/json" \
    -d '{
        "text": "你好，世界",
        "reference_audio": "path/to/reference.wav",
        "speed": 1.0
    }' \
    --output output.wav
```

#### 3. Same-language Voice Cloning
```bash
curl -X POST http://localhost:9233/v1/model/cosyvoice/clone_eq \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Hello world",
        "reference_audio": "path/to/reference.wav",
        "reference_text": "Reference text",
        "speed": 1.0
    }' \
    --output output.wav
```

## Configuration

- Default port: 9233
- Supported voice types:
  - Chinese Female/Male
  - English Female/Male
  - Japanese Male
  - Cantonese Female
  - Korean Female

## Environment Variables

- `RAY_ADDRESS`: Ray cluster address (optional)
- `PYTHONPATH`: Automatically configured by the application
- `CUDA_VISIBLE_DEVICES`: GPU device selection (optional)

## Project Structure

```
cosyvoice-ray-serve-api/
├── cosyvoice/
│   ├── api.py        # Main API implementation
│   └── utils/        # Utility functions
├── logs/             # Log files
├── tmp/              # Temporary files
├── requirements.txt  # Python dependencies
└── pretrained_models/ # Downloaded model files
```

## Troubleshooting

Common issues and solutions:

1. Sox compatibility issues:
   - Ensure Sox and its development packages are properly installed
   - Check system audio configurations

2. CUDA/GPU issues:
   - Verify CUDA installation with `nvidia-smi`
   - Check PyTorch CUDA compatibility

3. Model download issues:
   - Check network connectivity to ModelScope
   - Ensure sufficient disk space for models

## License

This project is licensed under the Apache 2.0 License. For more details, please refer to the [ LICENSE ](LICENSE) file.

## Acknowledgments

- CosyVoice models from ModelScope
- Ray Project for the serving framework
- FFmpeg and Sox for audio processing