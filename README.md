# Point-to-Explain

An interactive learning toy powered by computer vision and AI that explains anything you point at with a camera.

## Overview

Point-to-Explain runs on an NVIDIA Jetson Nano and uses a camera to identify objects in real-time. Simply point the camera at anything, and the system will detect what it's looking at and provide an explanation about it.

Perfect for curious minds of all ages!

## Hardware Requirements

- **NVIDIA Jetson Nano** (or compatible NVIDIA edge device)
- **Camera** (USB webcam or CSI camera module)
- Power supply for Jetson Nano
- (Optional) Display for visual feedback
- (Optional) Speaker for audio output

## Features

- Real-time object detection using camera feed
- AI-powered explanations of detected objects
- Optimized for edge computing on NVIDIA Jetson hardware
- Interactive and educational

## Getting Started

### Prerequisites

- NVIDIA Jetson Nano with JetPack SDK installed
- Python 3.x
- Camera drivers configured

### Installation

```bash
# Clone the repository
git clone https://github.com/ajfrai/point-to-explain.git
cd point-to-explain

# Install dependencies
pip install -r requirements.txt
```

### Testing the Camera

Test your camera connection with the included camera reader module:

```bash
# For CSI camera (default)
python3 camera_reader.py --type csi

# For USB camera
python3 camera_reader.py --type usb --id 0

# Custom resolution and framerate
python3 camera_reader.py --type csi --width 1920 --height 1080 --fps 30
```

Press 'q' to quit, 's' to save a snapshot.

### Usage

```bash
# Run the application (coming soon)
# python main.py
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

The project includes comprehensive unit tests for all modules:

```bash
# Run all tests
python3 test_camera_reader.py

# Or use pytest for more detailed output
pytest test_camera_reader.py -v

# Run tests with coverage report
pytest test_camera_reader.py --cov=camera_reader --cov-report=html
```

**Test Coverage:**
- Camera initialization (CSI and USB)
- Frame capture and reading
- Resource management and cleanup
- Context manager functionality
- Error handling
- GStreamer pipeline generation

All tests use mocking to avoid requiring actual camera hardware.

## Project Structure

```
point-to-explain/
├── camera_reader.py          # Camera interface module for CSI/USB cameras
├── test_camera_reader.py     # Unit tests for camera module
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## Architecture

This project will leverage:
- Computer vision for object detection
- AI/ML models for object recognition
- Natural language generation for explanations
- NVIDIA GPU acceleration for real-time performance

### Camera Module

The `camera_reader.py` module provides a unified interface for both CSI and USB cameras:
- **CSI Camera Support**: Uses GStreamer pipeline with hardware acceleration (nvarguscamerasrc)
- **USB Camera Support**: Standard OpenCV VideoCapture interface
- **Features**: Configurable resolution, framerate, and flip methods
- **Context Manager**: Easy resource management with Python's `with` statement

## Development Status

🚧 **Under Development** - This project is in early stages.

## Future Enhancements

- [ ] Multiple language support
- [ ] Voice output for explanations
- [ ] Learning mode to remember preferences
- [ ] Offline mode with cached explanations
- [ ] Custom object training

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
