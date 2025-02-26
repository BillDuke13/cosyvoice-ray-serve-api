# CosyVoice Ray Serve API Documentation

This document provides technical details about the implementation of the CosyVoice Ray Serve API.


## Overview

The CosyVoice Ray Serve API is built on top of the [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) framework, which provides a scalable model serving system. The API wraps the CosyVoice2 text-to-speech model and exposes it through HTTP endpoints for high-performance, scalable deployment with GPU acceleration and efficient resource management.

## Key Features

1. **Zero-shot Voice Cloning**: Clone any voice with a single audio sample
2. **Cross-lingual Synthesis**: Maintain voice identity across different languages
3. **Instruction-based Synthesis**: Control speech characteristics with natural language
4. **Fine-grained Control**: Adjust emotions, prosody, and speaking style
5. **Streaming Audio Generation**: Real-time audio synthesis for low-latency applications
6. **MP3 Format Support**: Efficient audio compression for smaller file sizes
7. **Reference Audio Caching**: Improved performance for repeated voice cloning requests

## Architecture

### Components

1. **Ray Serve Deployment**: The core of the API is implemented as a Ray Serve deployment class (`CosyVoiceService`).
2. **Audio Processor**: A utility class (`AudioProcessor`) for handling audio processing tasks.
3. **HTTP Endpoints**: RESTful endpoints for different TTS operations.
4. **GPU Resource Management**: Logic for detecting and utilizing available GPU resources.
5. **Logging System**: Comprehensive logging with file rotation.

### Ray Serve Integration

The API uses Ray Serve's deployment system to:

- Manage the lifecycle of the TTS service
- Handle HTTP requests and responses
- Allocate GPU resources efficiently
- Provide health monitoring
- Enable horizontal scaling

## Implementation Details

### CosyVoiceService Class

The `CosyVoiceService` class is the main deployment class that:

1. **Initialization** (`__init__`):
   - Detects and configures GPU usage if available
   - Sets up the CosyVoice2 model
   - Initializes the audio processor
   - Configures voice prompts
   - Registers a cleanup handler for resource management
   - Initializes reference audio cache

2. **Resource Management**:
   - `_cleanup_resources()`: Cleans up GPU memory and model references
   - `_setup_models()`: Initializes and loads the CosyVoice2 models
   - `_download_models()`: Downloads model files from ModelScope

3. **Request Handling** (`__call__`):
   - Processes different endpoint paths
   - Validates request parameters
   - Handles both streaming and non-streaming responses
   - Provides CORS support
   - Routes requests to appropriate handlers

4. **Parameter Validation** (`_validate_params`):
   - Ensures required parameters are present
   - Validates parameter types and values

5. **Voice Prompt Management** (`_get_prompt_path`):
   - Retrieves voice prompt audio files
   - Processes and caches reference audio
   - Handles different voice types

6. **Batch Processing** (`batch`):
   - Handles different TTS modes (standard, zero-shot, cross-lingual, instruction-based)
   - Processes audio generation requests
   - Manages file output and cleanup

### AudioProcessor Class

The `AudioProcessor` class provides utilities for:

1. **Reference Audio Processing** (`process_reference_audio`):
   - Converts input audio to a mono, 16-bit PCM WAV file
   - Uses FFmpeg for format conversion
   - Handles various input formats

2. **Audio Normalization** (`normalize_audio`):
   - Scales audio to the range [-1, 1]
   - Handles different tensor dimensions and data types

3. **Audio Validation** (`validate_audio`):
   - Checks if the input is a tensor
   - Verifies the tensor is not empty
   - Checks for NaN or infinite values

### HTTP Endpoints

The API exposes several HTTP endpoints under `/v1/model/cosyvoice`:

1. `/tts`: Standard text-to-speech with default or specified voice
2. `/zero_shot`: Zero-shot voice cloning with reference audio and text
3. `/cross_lingual`: Cross-lingual voice cloning with reference audio
4. `/instruct`: Instruction-based TTS for fine-grained control
5. `/healthcheck`: Health monitoring endpoint

Each endpoint supports:
- JSON request bodies
- Form data for file uploads
- Streaming responses (where applicable)
- Error handling with appropriate status codes

### GPU Resource Management

The API includes sophisticated GPU resource management:

1. **GPU Detection and Selection**:
   - Detects available GPUs using Ray's GPU allocation system
   - Maps Ray GPU IDs to CUDA device IDs
   - Sets the appropriate CUDA device for each replica

2. **Memory Optimization**:
   - Configures PyTorch CUDA memory allocation settings
   - Implements proper cleanup to release GPU memory
   - Provides CPU fallback if GPU initialization fails

3. **Error Handling**:
   - Gracefully handles GPU initialization failures
   - Logs detailed GPU information for debugging
   - Tests GPU access before model initialization

### Streaming Support

The API supports streaming audio generation:

1. **Real-time MP3 Streaming**:
   - Uses FFmpeg for on-the-fly PCM to MP3 conversion
   - Streams MP3 data chunks as they're generated
   - Provides proper MIME types and headers for browser compatibility
   - Handles errors during streaming with graceful cleanup

2. **Non-streaming Alternative**:
   - Collects all audio segments
   - Concatenates into a single response
   - Converts to MP3 format for efficient delivery
   - Validates the final audio file

## Code Structure

### Main Components

```
cosyvoice/api.py
├── AudioProcessor class
│   ├── process_reference_audio()
│   ├── normalize_audio()
│   └── validate_audio()
├── CosyVoiceService class
│   ├── __init__()
│   ├── _cleanup_resources()
│   ├── _setup_models()
│   ├── _download_models()
│   ├── __call__()
│   ├── _validate_params()
│   ├── _get_prompt_path()
│   └── batch()
└── main()
```

### Key Functions

1. `__init__()`: Initializes the service, detects GPUs, and sets up models
2. `_setup_models()`: Downloads and initializes the CosyVoice model
3. `__call__()`: Handles HTTP requests and routes to appropriate handlers
4. `batch()`: Processes TTS requests in batch mode
5. `_get_prompt_path()`: Retrieves and caches voice prompt audio
6. `main()`: Entry point that starts the Ray Serve service

## API Endpoints in Detail

### 1. Standard TTS (`/tts`)

Synthesizes speech from text using a specified voice type.

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",  // Required
    "voice_type": "qwen",          // Optional (default: "qwen")
    "speed": 1.0,                  // Optional (default: 1.0)
    "stream": false                // Optional, enable streaming output (default: false)
}
```

**Response:**
- For non-streaming requests:
  - Content-Type: `audio/mpeg`
  - Body: MP3 audio file (24kHz sample rate)
- For streaming requests:
  - Content-Type: `audio/mpeg`
  - Body: Chunked MP3 audio stream

### 2. Zero-shot Voice Cloning (`/zero_shot`)

Clones a voice from a reference audio file and uses it to speak text.

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",           // Required
    "reference_audio": "path/to/audio.wav", // Required
    "reference_text": "Reference text",     // Required
    "speed": 1.0,                           // Optional (default: 1.0)
    "stream": false                         // Optional, enable streaming output (default: false)
}
```

**Response:**
- Content-Type: `audio/mpeg`
- Body: MP3 audio file (24kHz sample rate)

### 3. Cross-lingual Voice Cloning (`/cross_lingual`)

Clones a voice from a reference audio file and uses it to speak text in any supported language.

**Method:** `POST`

**Request Body:**
```json
{
    "text": "Text to synthesize",           // Required
    "reference_audio": "path/to/audio.wav", // Required
    "speed": 1.0,                           // Optional (default: 1.0)
    "stream": false                         // Optional, enable streaming output (default: false)
}
```

**Response:**
- Content-Type: `audio/mpeg`
- Body: MP3 audio file (24kHz sample rate)

### 4. Instruction-based TTS (`/instruct`)

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
- Content-Type: `audio/mpeg`
- Body: MP3 audio file (24kHz sample rate)

### 5. Health Check (`/healthcheck`)

Checks if the service is running properly.

**Method:** `POST`

**Response:**
- Content-Type: `application/json`
- Body: `{"status": "healthy"}`

## Performance Optimizations

1. **Reference Audio Caching**: Processed reference audio is cached in `_reference_audio_cache` to avoid redundant processing
2. **GPU Memory Management**: Optimized CUDA memory allocation settings
3. **MP3 Compression**: Reduces bandwidth usage and storage requirements
4. **Streaming Generation**: Enables real-time audio generation for low-latency applications
5. **Horizontal Scaling**: Multiple replicas can be deployed for high-throughput scenarios
6. **Graceful Fallback**: Falls back to CPU if GPU initialization fails
7. **Efficient Audio Processing**: Keeps processing on GPU as much as possible before final CPU conversion

## Error Handling

The API implements comprehensive error handling:

1. Request validation with detailed error messages
2. Audio processing error detection and reporting
3. Model initialization failure handling
4. GPU resource allocation error recovery
5. Proper HTTP status codes for different error types

## Logging

The logging system provides:

1. Detailed logs with timestamps and log levels
2. Log rotation with compression
3. Separate log files for application logs
4. Console output for immediate feedback
5. GPU resource usage logging

## Security Considerations

1. **CORS Support**: Configurable CORS headers for cross-origin requests
2. **Input Validation**: Thorough validation of all request parameters
3. **Resource Limits**: Configurable limits on request processing
4. **Error Information**: Careful control of error information exposure

## Audio Processing Details

The API implements a sophisticated audio processing pipeline:

1. **Generation**: Audio is generated by the CosyVoice2 model in chunks
2. **Normalization**: Audio is normalized to the [-1, 1] range
3. **Conversion**: Float32 audio is converted to int16 PCM format
4. **MP3 Encoding**: FFmpeg is used to convert PCM to MP3 format
5. **Validation**: Audio is validated to ensure quality and prevent errors
6. **Streaming**: For streaming requests, audio is processed and streamed in real-time

## Future Improvements

Potential areas for enhancement:

1. **Authentication**: Add API key or OAuth-based authentication
2. **Rate Limiting**: Implement request rate limiting
3. **Metrics Collection**: Add Prometheus metrics for monitoring
4. **Model Versioning**: Support multiple model versions
5. **Dynamic Scaling**: Implement auto-scaling based on load
6. **Audio Format Options**: Allow clients to specify desired audio format
7. **Batch Processing**: Support for processing multiple TTS requests in a single API call
