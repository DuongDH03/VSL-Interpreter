# VSL-Interpreter: Using ST-GCN++ for Sign Language Recognition

This folder contains the server-side code for the Visual Sign Language (VSL) Interpreter application.

## Requirements

The application requires several dependencies, particularly for the ST-GCN++ model:
- **Python 3.8** (strongly recommended for compatibility)
- PyTorch (CPU version)
- MMCV
- PYSKL (for ST-GCN++ implementation)
- MediaPipe
- Node.js (for the server)

## Directory Structure

- `models/`: Contains model configuration files and weights
  - `configs/`: Configuration files for ST-GCN++ model
  - `stgcn++_model.pth`: Pre-trained model weights
  - `class_map.json`: Mapping from class indices to sign labels
- `utils/`: Utility functions for keypoint processing and model setup
- `controllers/`: Node.js controllers for handling API requests
- `routes/`: API route definitions
- `workers/`: Python worker scripts for efficient model inference
  - `pyskl_inference_worker.py`: Long-running worker for continuous inference
- `stgcn_inference.py`: Legacy script for inference (now used by worker process)

## New Worker-Based Architecture

This application now uses a more efficient worker-based architecture:
1. **One-time model loading**: The Python worker loads the model once when the server starts
2. **Persistent process**: The worker stays running, accepting inference requests via stdin/stdout
3. **Memory efficiency**: Avoids repeatedly loading the model for each request
4. **Better performance**: Drastically reduces latency for real-time sign language translation

## Setting Up

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Python Environment Setup Instructions:

   ### Step 1: Create a Python 3.8 environment
   
   **Option A: Using Conda** (recommended)
   ```powershell
   # Create new environment with Python 3.8
   conda create -n vsl-env python=3.8
   
   # Activate the environment
   conda activate vsl-env
   ```
   
   **Option B: Using Python venv** (if Python 3.8 is installed)
   ```powershell
   # Create virtual environment
   python -m venv vsl-env
   
   # Activate on Windows
   .\vsl-env\Scripts\activate
   
   # Activate on Linux/Mac
   source vsl-env/bin/activate
   ```   ### Step 2: Install Dependencies

   **Method 1: Automatic Setup Script** (recommended)
   ```powershell
   # Run the setup script to install all required packages
   python utils/setup_pyskl.py
   ```

   **Method 2: Manual Installation**
   ```powershell
   # Install PyTorch (CPU version to avoid memory issues)
   pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
   
   # Install basic requirements
   pip install -r requirements.txt
   
   # Install MMCV
   pip install mmcv-full==1.5.0
   
   # Install MM series packages
   pip install mmdet==2.23.0 mmpose==0.24.0
   
   # Install PYSKL from source if available
   # If you have cloned the PYSKL repository:
   cd path/to/pyskl
   pip install -e .
   ```

3. Check if environment is correctly set up:
   ```powershell
   python check_pyskl.py
   ```

4. Start the server:
   ```powershell
   node server.js
   ```

## API Endpoints

- `POST /api/translate/live-video`: Process live video stream from webcam
- `POST /api/translate/video`: Process uploaded video file
- `POST /api/translate/image`: Process static image

## Testing the Worker Directly

You can test the worker process directly without starting the full server:

```powershell
node test_worker.js
```

This will initialize the Python worker, send some test data, and display the results.

## Troubleshooting

### PyTorch CUDA Issues

If you encounter CUDA-related errors like "The paging file is too small for this operation", use our CPU-only configuration:

1. Make sure PyTorch is installed with CPU support only:
   ```powershell
   pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
   ```

2. Verify that PyTorch is using CPU:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   # Should print "CUDA available: False"
   ```

### Configuration File Not Found

If you get "Config file not found" errors:
1. Make sure the config file is in the correct location: `/models/configs/stgcnpp_hand27.py`
2. Verify that your current working directory is the server directory when running the script

### Worker Communication Errors

If the worker is not responding:
1. Check the worker process is running: `ps aux | grep pyskl_inference_worker.py`
2. Look for error messages in the server logs
3. Try running the test script: `node test_worker.js`

### Missing Model Weights

If the model weights aren't found:
1. Ensure `stgcn++_model.pth` is in the `models/` directory
2. If using a custom model, update the file path in `workers/pyskl_inference_worker.py`

## Development Notes

- The ST-GCN++ model expects keypoints in the format (C,T,V,M) where:
  - C: channels (x,y,confidence), usually 3
  - T: frames/time steps
  - V: vertices/joints
  - M: number of people (usually 1)
  
- Our keypoint processing workflow:
  1. Extract MediaPipe hand and pose landmarks from video frames
  2. Convert to ST-GCN++ format with `extract_keypoints()`
  3. Buffer frames until we have enough for inference with `KeypointBuffer`
  4. Run model inference on the collected sequence

- The Python worker process maintains its own buffer, allowing for seamless real-time inference without having to send the entire buffer from Node.js for each frame.
- 9 keypoints for each hand

## Troubleshooting

If you encounter issues with the model:

1. Verify that all dependencies are installed correctly
2. Check that model weights and config files are present
3. Ensure the input data format matches what the model expects
4. Look for error logs in the server console

### Common Issues

#### PyTorch CUDA Error
If you see this error:
```
OSError: [WinError 1455] The paging file is too small for this operation to complete. 
Error loading "...\torch\lib\cudnn_cnn_infer64_8.dll" or one of its dependencies.
```

This means PyTorch with CUDA support is trying to allocate memory but failing. Use the CPU version instead:
```powershell
pip uninstall torch torchvision -y
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
```

#### Config Import Error
If you see config import errors, make sure the correct directory structure is maintained:
```
server/
  ├── models/
  │   ├── configs/
  │   │   └── stgcnpp_hand27.py
  │   └── stgcn++_model.pth
  └── stgcn_inference.py
```

For more information about the ST-GCN++ model and PYSKL, visit:
https://github.com/kennymckormick/pyskl
