# CosyVoice Ray Serve API

This project provides a Ray Serve-based HTTP API for the [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) Text-to-Speech (TTS) system. It enables scalable and robust deployment of CosyVoice, offering features like RESTful API endpoints, Docker containerization, health monitoring, streaming audio generation, and efficient resource management.

## Project Overview

The CosyVoice Ray Serve API extends the capabilities of the original CosyVoice project by wrapping it in a production-ready API service. This service is built using Ray Serve, which handles deployment, scaling, and automatic resource allocation (including GPU detection). The implementation emphasizes:

- **Maintainability**: Clean, well-structured code with clear separation of concerns.
- **Performance**: Optimized for speed, including efficient GPU utilization and reference audio caching.
- **Reliability**: Comprehensive error handling, logging, and robust resource management.
- **Scalability**: Leverages Ray Serve for automatic scaling of TTS worker replicas.

## Core Features (from CosyVoice)

This API exposes the powerful TTS capabilities of the underlying CosyVoice model:

- **Zero-shot Voice Cloning**: Replicate any voice from a short audio sample (e.g., 3-10 seconds).
- **Cross-lingual Speech Synthesis**: Maintain a consistent voice identity when synthesizing speech in different languages.
- **Instruction-based Speech Synthesis**: Control speech characteristics (emotion, prosody, style) using natural language prompts.
- **Fine-grained Control**: Adjust various speech parameters for nuanced output.
- **Multi-language Support**:
  - Chinese (Mandarin)
  - English
  - Japanese
  - Cantonese
  - Korean
- **Diverse Voice Types**: Supports a range of male and female voice profiles.
- **Adjustable Speech Speed**: Control the rate of speech generation.
- **Streaming Audio Output**: Enables real-time audio synthesis for applications requiring low latency.
- **MP3 Format Support**: Delivers audio in the efficient MP3 format.

## Getting Started

Follow these instructions to set up and run the CosyVoice Ray Serve API.

### Prerequisites

- Python 3.11 (recommended)
- FFmpeg (for audio processing)
- Sox and its development packages (for audio processing)
- CUDA 11.7 or newer (optional, for GPU acceleration)
- Ray (core serving framework)
- PyTorch 2.0 or newer
- Torchaudio (for audio I/O)

### Installation Options

#### Option 1: Local Installation (Recommended for Development)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BillDuke13/cosyvoice-ray-serve-api.git
   cd cosyvoice-ray-serve-api
   ```

2. **Create and activate a Conda environment:**

   ```bash
   conda create -n cosyvoice-api python=3.11 -y
   conda activate cosyvoice-api
   ```

3. **Install system dependencies:**

   - **For Ubuntu/Debian:**

     ```bash
     sudo apt-get update
     sudo apt-get install -y ffmpeg sox libsox-fmt-all libsox-dev
     ```

   - **For CentOS/RHEL:**

     ```bash
     sudo yum install -y epel-release
     sudo yum install -y ffmpeg sox sox-devel
     ```

   - **For macOS (using Homebrew):**

     ```bash
     brew install ffmpeg sox
     ```

4. **Install Python dependencies:**

   ```bash
   conda install -y -c conda-forge pynini==2.1.5 
   pip install -r requirements.txt
   ```

5. **Model Download:**
   Models are typically downloaded automatically by the `api.py` script on the first run if they are not found in the `pretrained_models` directory.

#### Option 2: Docker Installation (Recommended for Production & Easy Deployment)

1. **Clone the repository:** (If not already done)

   ```bash
   git clone https://github.com/BillDuke13/cosyvoice-ray-serve-api.git
   cd cosyvoice-ray-serve-api
   ```

2. **Build the Docker image:**

   ```bash
   docker build -t cosyvoice-api:latest .
   ```

   This command uses the provided `Dockerfile` to create a container image with all necessary dependencies and the application code.

### Running the Application

#### Using Docker

1. **Run the Docker container:**

   - **Without GPU support:**

     ```bash
     docker run -d -p 8000:8000 --name cosyvoice-container cosyvoice-api:latest
     ```

   - **With GPU support (NVIDIA Docker Toolkit required):**

     ```bash
     docker run -d --gpus all -p 8000:8000 --name cosyvoice-container cosyvoice-api:latest
     ```

     Ensure you have the NVIDIA Docker Toolkit installed and configured on your host machine.

2. **Accessing the API:**
   The API will be available at `http://localhost:8000`.

#### Locally (without Docker)

1. **Ensure all dependencies from Option 1 are installed and the Conda environment is active.**

2. **Start the Ray Serve application:**
   Navigate to the project's root directory in your terminal and run:

   ```bash
   serve run api:cosyvoice_app
   ```

   This command starts a local Ray cluster and deploys the `CosyVoiceService` defined in `api.py`.

3. **Accessing the API:**
   The API will be available at `http://localhost:8000`.

## API Endpoints

The service exposes the following HTTP endpoints:

- **`POST /tts`**: Standard Text-to-Speech.
  - **Request Body (JSON):**

    ```json
    {
        "text": "Hello, this is a test.",
        "voice_type": "qwen", // Optional, defaults to "qwen" or a configured default
        "speed": 1.0,        // Optional, speech speed factor (Note: effect may vary by model method)
        "stream": false      // Optional, set to true for streaming response
    }
    ```

  - **Response:** MP3 audio file or a streaming audio response if `stream` is true.

- **`POST /zero_shot_tts`**: Zero-shot voice cloning.
  - **Request Body (form-data):**
    - `text`: The text to synthesize.
    - `reference_text`: The transcript of the reference audio.
    - `reference_audio`: The reference audio file (e.g., WAV, MP3).
    - `speed` (optional): Speech speed factor.
    - `stream` (optional): Set to true for streaming response.
  - **Response:** MP3 audio file or a streaming audio response.

- **`POST /cross_lingual_tts`**: Cross-lingual voice synthesis.
  - **Request Body (form-data):**
    - `text`: The text to synthesize (can be in a different language than the reference).
    - `reference_audio`: The reference audio file.
    - `speed` (optional): Speech speed factor.
    - `stream` (optional): Set to true for streaming response.
  - **Response:** MP3 audio file or a streaming audio response.

- **`POST /instruct_tts`**: Instruction-based speech synthesis.
  - **Request Body (JSON):**

    ```json
    {
        "text": "Can you say this with a happy tone?",
        "instruction": "Speak happily and energetically.",
        "voice_type": "qwen", // Optional
        "speed": 1.0         // Optional
    }
    ```

  - **Response:** MP3 audio file.

- **`GET /health`**: Health check endpoint.
  - **Response:**

    ```json
    {"status": "healthy", "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff"}
    ```

- **`GET /config`**: Get current service configuration.
  - **Response:** JSON object with current settings (e.g., default voice, available prompts).

## Configuration

Key configurations can be adjusted via environment variables (especially for Docker deployment) or directly in `api.py`:

- `MAX_ONGOING_REQUESTS_PER_REPLICA`: Max concurrent requests a single replica handles.
- `MAX_QUEUED_REQUESTS_DEPLOYMENT`: Max requests to queue at the deployment level.
- `MIN_REPLICAS`, `INITIAL_REPLICAS`, `MAX_REPLICAS`: Control autoscaling behavior.
- `TARGET_ONGOING_REQUESTS`: Target ongoing requests for autoscaling decisions.
- `NUM_GPUS_PER_REPLICA`: Number of GPUs to assign per replica (e.g., 0 or 1).
- `CUDA_VISIBLE_DEVICES`: Standard NVIDIA environment variable to control GPU visibility.

Default voice prompts are defined in `VOICE_PROMPTS` in `api.py`. New `.wav` files can be added to the `asset/` directory and referenced there.

## Project Structure

```
cosyvoice-ray-serve-api/
├── api.py                # Main Ray Serve application, API logic, and service class
├── Dockerfile            # For building the Docker container
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── asset/                # Directory for voice prompt audio files (e.g., qwen.wav)
│   └── qwen.wav
├── cosyvoice/            # Submodule or copied source of the original CosyVoice project
│   ├── bin/
│   ├── cli/
│   └── ... (other CosyVoice directories)
├── pretrained_models/    # Directory where models are downloaded/stored
│   └── CosyVoice2-0.5B/
├── tmp/                  # Temporary file storage (created at runtime)
└── logs/                 # Log file storage (created at runtime)
```

## Development and Customization

- **Adding Voice Prompts**: Place new `.wav` files in the `asset/` directory and update the `VOICE_PROMPTS` dictionary in `api.py`.
- **Modifying Model Behavior**: Adjustments to the core TTS model (`CosyVoice2`) can be made within its source code in the `cosyvoice/` directory.
- **Scaling Configuration**: Tweak autoscaling parameters in the `@serve.deployment` decorator in `api.py` to suit your load profile.

## Logging and Monitoring

- Logs are output to `stdout` (visible with `docker logs <container_id>`) and also saved to files in the `logs/` directory within the container (or locally if not using Docker).
- The `/health` endpoint can be used for basic liveness and readiness checks in orchestration systems like Kubernetes.

## Troubleshooting

- **Model Download Issues**: Ensure you have internet connectivity when running for the first time (or during Docker build if pre-downloading models). Check permissions for the `pretrained_models/` directory.
- **FFmpeg/Sox Not Found**: Verify that these system dependencies are correctly installed and accessible in the system's PATH (or within the Docker container).
- **CUDA Errors**:
  - Ensure your NVIDIA drivers and CUDA toolkit version on the host are compatible with the PyTorch version used.
  - If using Docker, confirm the NVIDIA Docker Toolkit is installed and working.
  - Check `CUDA_VISIBLE_DEVICES` settings.
- **Pynini Installation**: Pynini can be tricky. If `pip install -r requirements.txt` fails on Pynini, you might need to install it separately using Conda (as shown in local setup) or find a wheel/build method suitable for your OS/architecture. The Dockerfile attempts a standard pip install; if this fails, the Docker build process for Pynini might need adjustment.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements. Ensure that your contributions align with the project's coding standards and include relevant tests or documentation updates.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details. The underlying CosyVoice project may have its own licensing terms, which should also be respected.

## Acknowledgements

- The [FunAudioLLM team](https://github.com/FunAudioLLM) for creating and open-sourcing CosyVoice.
- The [Ray Serve team](https://docs.ray.io/en/latest/serve/index.html) for the powerful serving framework.
