# facemaker-fm1

A comprehensive toolkit for hardware-accelerated face swapping in images and videos using TensorFlow and Torch machine learning frameworks.

## Features

- 🖼️ **Image Face Swapping**: Swap faces in images with high quality results
- 🎥 **Video Face Swapping**: Process videos with frame-by-frame face swapping
- 🔍 **Face Detection**: Detect and extract faces from images
- ✨ **Face Enhancement**: Optional GFPGAN-based face restoration
- 🌐 **Multiple Interfaces**:
  - Web UI (Gradio)
  - REST API (Flask)
  - Python Library

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ikmalsaid/facemaker-fm1.git
cd facemaker-fm1
```

2. Create a Python virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install base requirements:
```bash
pip install -r requirements_base.txt
```

4. Install CUDA requirements (for GPU acceleration):
```bash
pip install -r requirements_cuda.txt
```

## Usage

### Python Library

```python
from main import FacemakerFM1

# Initialize
fm = FacemakerFM1()

# Swap faces in image
result = fm.swap_image(
    target_image="path/to/target.jpg",
    source_image="path/to/source.jpg",
    target_index=0,  # Target face index
    source_index=0,  # Source face index
    swap_all=False,  # Swap all faces in target
    face_restore=True,  # Enable GFPGAN enhancement
    face_restore_model="GFPGAN 1.3"  # GFPGAN model version
)

# Swap faces in video
result = fm.swap_video(
    source_image="path/to/source.jpg",
    target_video="path/to/video.mp4",
    source_index=0,
    face_restore=True,
    face_restore_model="GFPGAN 1.3"
)

# Detect faces
faces, count, index = fm.detect_faces("path/to/image.jpg")
```

### Web UI

Start the Gradio web interface:

```python
fm = FacemakerFM1(mode='webui')
# OR
fm.start_webui(
    host="0.0.0.0",
    port=3225,
    browser=True,
    upload_size="10MB",
    public=False
)
```

### REST API

Start the Flask API server:

```python
fm = FacemakerFM1(mode='api')
# OR
fm.start_api(
    host="0.0.0.0",
    port=3223,
    debug=False
)
```

#### API Endpoints

- `POST /api/swap/image`: Swap faces in image
- `POST /api/swap/video`: Swap faces in video
- `POST /api/detect`: Detect faces in image
- `GET /api/download/<filename>`: Download processed files

## License

See [LICENSE](LICENSE) for details.
