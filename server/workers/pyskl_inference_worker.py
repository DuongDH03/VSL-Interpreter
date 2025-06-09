#!/usr/bin/env python
# filepath: d:\VSL-Interpreter\server\workers\pyskl_inference_worker.py

import sys
import json
import os
import numpy as np
import traceback
from collections import deque
import importlib.util
import time
import io

# --- Path Setup (CRUCIAL) ---
# Assuming this script is in `your_project/server/workers/pyskl_inference_worker.py`
# and your PySKL repo is at `your_project/pyskl_repo` or installed in the venv.
# And your configs/models are relative to `your_project/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up 3 levels
sys.path.insert(0, PROJECT_ROOT) # Add project root to path for relative imports

# --- PySKL Availability Check and Import ---
PYSKL_AVAILABLE = False
try:
    from mmcv import Config
    
    pyskl_repo_path = os.path.join(PROJECT_ROOT, 'server', 'pyskl_repo') # Adjust if pyskl is named differently
    if os.path.exists(pyskl_repo_path) and os.path.isdir(pyskl_repo_path):
        PYSKL_AVAILABLE = True
        sys.path.insert(0, pyskl_repo_path)
        sys.stderr.write(f"Worker: Found PySKL repo at {pyskl_repo_path}, adding to path\n")

        from pyskl.apis.inference import init_recognizer
        from pyskl.datasets.pipelines import Compose
        sys.stderr.write("Worker: Successfully imported PySKL functions from local repository\n")
    
    # # Re-check after path adjustment
    # if importlib.util.find_spec('pyskl'):
    #     PYSKL_AVAILABLE = True
    #     sys.stderr.write("Worker: PySKL and essential dependencies found.\n")
    # else:
    #     raise ImportError("PySKL module not found after path attempts.")

except ImportError as e:
    sys.stderr.write(f"Worker: PySKL or its core dependencies not found: {e}\n")
    sys.stderr.write("Worker: Falling back to placeholder model. Please ensure PySKL, MMCV etc. are correctly installed in this Python environment.\n")
    sys.stderr.write("Hint: pip install mmcv-full==1.6.0 torchvision==0.13.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html (adjust for your env)\n")
    sys.stderr.write("Then install pyskl: pip install -e . from pyskl repo.\n")
    
# --- PyTorch Availability Check ---
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    sys.stderr.write("Worker: PyTorch loaded successfully.\n")
except (ImportError, OSError) as e:
    sys.stderr.write(f"Worker: Warning: PyTorch could not be loaded: {e}\n")
    sys.stderr.write("Worker: Using placeholder inference without machine learning model.\n")

# --- Import our custom utilities (assuming they are in PROJECT_ROOT/utils) ---
try:
    # Add server directory to path
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, server_dir)
    
    # Now import from server/utils
    from utils.keypoint_extractor import KeypointBuffer
    sys.stderr.write(f"Worker: Successfully imported KeypointBuffer from {server_dir}/utils/keypoint_extractor.py\n")
except ImportError as e:
    sys.stderr.write(f"Worker: Could not import KeypointBuffer from utils.keypoint_extractor: {e}\n")
    sys.stderr.write(f"Current import search paths: {sys.path}\n")
    sys.exit(1) # Cannot proceed without KeypointBuffer

# --- Configuration Values (derived from your demo/config) ---
# IMPORTANT: Adjust these paths relative to PROJECT_ROOT
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'server', 'models', 'configs', 'stgcnpp_hand27.py')
CHECKPOINT_FILE = os.path.join(PROJECT_ROOT, 'server', 'models', 'stgcn++_model.pth') # Standard PySKL checkpoint path
LABEL_MAP_FILE = os.path.join(PROJECT_ROOT, 'server', 'model', 'label_name.txt')

LAYOUT = 'hand_body_27' # Must match layout for extractKeypointsJs in frontend
DEVICE = 'cpu' # Use 'cuda' if GPU is available and configured
NUM_FRAMES = 100 # Default window size (clip_len) from stgcnpp_hand27.py config
NUM_JOINTS = 27 # Number of joints in hand_body_27 layout
SLIDING_WINDOW_STEP = 8 # Default step size, for future use if worker supports sliding
PREDICT_PER_NFRAME = 3 # Predict every N frames to reduce load, from your demo

# --- Global variables for model, pipeline, and buffer ---
recognizer = None
test_pipeline_instance = None # Renamed to avoid conflict with imported Compose
label_names = []
keypoint_buffer_instance = KeypointBuffer(buffer_size=NUM_FRAMES, step_size=SLIDING_WINDOW_STEP)
frame_idx_counter = 0
last_predictions_deque = deque(maxlen=5) # For smoothing, renamed to avoid conflict

# --- Class Map Loading ---
CLASS_MAP = {}
try:
    CLASS_MAP_PATH = os.path.join(PROJECT_ROOT, 'server', 'models', 'class_map.json') # Adjust path if different
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, 'r') as f:
            CLASS_MAP = json.load(f)
        sys.stderr.write(f"Worker: Loaded class mapping with {len(CLASS_MAP)} classes from {CLASS_MAP_PATH}\n")
    else:
        sys.stderr.write(f"Worker: Warning: Class map file not found at {CLASS_MAP_PATH}. Will use default class names.\n")
        # Default if not found (will be overridden by model config if available)
        CLASS_MAP = {str(i): f"sign_{i}" for i in range(10)} 
except Exception as e:
    sys.stderr.write(f"Worker: Error loading class map: {e}. Using default class names.\n")
    CLASS_MAP = {str(i): f"sign_{i}" for i in range(10)}

def get_class_names():
    return label_names if label_names else CLASS_MAP # Prefer loaded label_names, fallback to CLASS_MAP

# --- Model Loading Function ---
def load_model():
    global recognizer, test_pipeline_instance, label_names, NUM_FRAMES

    if not TORCH_AVAILABLE:
        sys.stderr.write("Worker: PyTorch not available, skipping model load\n")
        return False # Indicate failure

    try:
        if PYSKL_AVAILABLE:
            sys.stderr.write(f"Worker: Attempting to load PySKL model from config: {CONFIG_FILE}\n")
            if not os.path.exists(CONFIG_FILE):
                raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                config = Config.fromfile(CONFIG_FILE)

                # Initialize recognizer
                recognizer = init_recognizer(config, CHECKPOINT_FILE, DEVICE)
            finally:
                # Restore original stdout and print captured output to stderr
                captured_output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                if captured_output:
                    sys.stderr.write("Worker: Captured stdout during model load:\n")
                    sys.stderr.write(captured_output) # Print captured output to stderr

            recognizer.eval()
            sys.stderr.write(f"Worker: PySKL recognizer initialized from {CHECKPOINT_FILE}\n")
            # Initialize test pipeline from config
            # Ensure the test_pipeline is compatible with directly provided keypoints
            # It should include steps like `FormatGCNInput` and `ToTensor`
            test_pipeline_instance = Compose(config.test_pipeline)
            sys.stderr.write("Worker: PySKL test pipeline composed.\n")
            # Load label names (prefer labels from config if available, otherwise use global CLASS_MAP)
            if hasattr(config.model.cls_head, 'num_classes'):
                # Override default CLASS_MAP if num_classes from config is available
                label_names = [f"Class {i}" for i in range(config.model.cls_head.num_classes)]
            # Attempt to load from label map file first
            loaded_labels = [CLASS_MAP.get(str(i), f"Class {i}") for i in range(len(CLASS_MAP))]
            if loaded_labels:
                label_names = loaded_labels
                sys.stderr.write(f"Worker: Loaded {len(label_names)} labels from label map file.\n")
            elif not label_names: # Fallback if config didn't provide num_classes and label map failed
                sys.stderr.write("Worker: Could not load labels from file or config. Defaulting to 10 classes.\n")
                label_names = [f"Class {i}" for i in range(10)]
            sys.stderr.write("Worker: Real PySKL model and pipeline loaded.\n")
            return True # Indicate success
        else:
            sys.stderr.write("Worker: PySKL not available, falling back to placeholder model.\n")
            return False
        
    except Exception as e:
        sys.stderr.write(f"Worker: Error loading model: {e}\n")
        sys.stderr.write(traceback.format_exc(file=sys.stderr))
        recognizer = None
        test_pipeline_instance = None
        return False # Indicate failure

# --- Inference Function ---
def run_inference():
    """
    Run inference on the buffered keypoints.
    This function will be called when `keypoint_buffer_instance` has enough frames.
    
    Returns:
        dict: Results of inference with predictions and confidences
    """
    global recognizer, test_pipeline_instance, last_predictions_deque, keypoint_buffer_instance

    if recognizer is None or not TORCH_AVAILABLE:
        return {
            "message": "Model not loaded or PyTorch not available",
            "prediction": "Not Ready",
            "label_index": -1,
            "score": 0.0
        }
    
    # Get the sequence from the buffer (it will pad if not enough frames)
    # keypoint_buffer_instance.get_input() returns data in shape (C, T, V) based on its extract_keypoints output
    # or (T, V, C) depending on your KeypointBuffer implementation.
    # Let's assume KeypointBuffer.get_input() provides (T, V, C) for simplicity matching frontend
    buffered_sequence_np = keypoint_buffer_instance.get_input()
    
    if buffered_sequence_np.shape[0] < NUM_FRAMES:
        return {
            "message": f"Buffering frames... ({buffered_sequence_np.shape[0]}/{NUM_FRAMES})",
            "prediction": "Buffering",
            "label_index": -1,
            "score": 0.0
        }

    # Prepare input for PySKL's test_pipeline
    # PySKL's pipeline usually expects a dict with 'keypoint' and other meta_keys
    # The 'keypoint' value for the pipeline typically is (M, T, V, C) for one person.
    # So if buffered_sequence_np is (T, V, C), we need to add the M=1 dimension
    
    # Create the dictionary that the PySKL test_pipeline expects
    # From `create_fake_anno` in original demo, keypoint is (M, T, V, C)
    # The frontend extracts (V, C) per frame. 
    # Convert from (T, V, C) to (M, T, V, C) where M=1 person for the pipeline input
    input_for_pipeline = np.expand_dims(buffered_sequence_np, axis=0) # Shape (1, T, V, C)

    # The `total_frames` used in the dummy annotation should be `NUM_FRAMES`
    fake_anno = dict(
        keypoint=input_for_pipeline,
        total_frames=NUM_FRAMES, # This should be the window size
        frame_dir='NA',
        label=-1, # Dummy label for inference
        start_index=0,
        modality='Pose', # Or 'Pose' depending on how PySKL handles unified input
        test_mode=True
    )
    
    # Apply test pipeline if it exists (for PySKL model)
    if test_pipeline_instance:
        sys.stderr.write(f"Worker: Input shape before pipeline: {input_for_pipeline.shape}\n")

        data_processed_by_pipeline = test_pipeline_instance(fake_anno)
        # The pipeline should output a dict with 'keypoint' tensor in (C, T, V, M) format
        input_tensor_for_model = data_processed_by_pipeline['keypoint']

        sys.stderr.write(f"Worker: Tensor shape after pipeline: {input_tensor_for_model.shape}\n")


        if input_tensor_for_model.ndim == 4:
            # Add the missing person dimension (M=1)
            # If shape is (N, C, T, V) we need to make it (N, C, T, V, M)
            input_tensor_for_model = input_tensor_for_model.unsqueeze(0)
            sys.stderr.write(f"Worker: Added missing dimension. New shape: {input_tensor_for_model.shape}\n")
        elif input_tensor_for_model.ndim == 3: # Expected: (C, T, V) if M is squeezed by pipeline
             # Add N and M=1 dimensions
             input_tensor_for_model = input_tensor_for_model.unsqueeze(0).unsqueeze(-1) 
             sys.stderr.write(f"Worker Debug: Tensor shape after unsqueeze(0) and unsqueeze(-1) for N, M: {input_tensor_for_model.shape}\n")
    else: # For placeholder model, manual tensor creation
        # Placeholder expects (N, C, T, V, M)
        # Convert (1, T, V, C) to (1, C, T, V, 1) or (N, C, T, V, M)
        # If input_for_pipeline is (1, T, V, C)
        input_tensor_for_model = torch.tensor(input_for_pipeline, dtype=torch.float32).permute(0, 3, 1, 2) # -> (1, C, T, V)
        input_tensor_for_model = input_tensor_for_model.unsqueeze(-1) # -> (1, C, T, V, 1) M=1
    
    device = next(recognizer.parameters()).device
    input_tensor_for_model = input_tensor_for_model[None].to(device)

    # Debug the tensor shape
    sys.stderr.write(f"Worker: Final input tensor shape: {input_tensor_for_model.shape}\n")

    # Forward pass
    with torch.no_grad():
        output = recognizer(input_tensor_for_model, return_loss=False) # Assuming batch size 1, get first element

        # Get the first element (assuming batch size 1)
        # The output shape should be [batch_size, num_classes]
        pred_scores = output[0]
        
        sys.stderr.write(f"Worker: Prediction scores shape: {pred_scores.shape}\n")
        
        # Convert to numpy for processing
        pred_scores_np = pred_scores
        
        # Get the predicted label (class with max score)
        pred_label = int(np.argmax(pred_scores_np))
        pred_score = float(pred_scores_np[pred_label])
        
        class_names = get_class_names()

        prediction = "Initializing..."

        # Apply confidence threshold like the original
        if pred_score < 0.3:
            prediction = "Unknown"
        else:
            prediction = class_names[pred_label] if pred_label < len(class_names) else f"unknown_{pred_label}"
        
        # Update last predictions for smoothing - store just the class name without score
        if pred_score > 0.3:
            class_name = class_names[pred_label] if pred_label < len(class_names) else f"unknown_{pred_label}"
            last_predictions_deque.append(class_name)


    # Apply temporal smoothing
    smoothed_prediction = "unknown"
    if last_predictions_deque:
        from collections import Counter
        prediction_counts = Counter(last_predictions_deque)
        most_common_entry = prediction_counts.most_common(1)[0]
        smoothed_prediction = most_common_entry[0]
        
    return {
        "message": "Inference complete",
        "prediction": prediction,
        "smoothed_prediction": smoothed_prediction,  # Smoothed over time
        "prediction": smoothed_prediction,
        "score": float(pred_score),
        "label_index": int(pred_label) # Return index of top raw prediction
    }

# --- Main Worker Loop ---
def main():
    global model, keypoint_buffer_instance, frame_idx_counter

    sys.stderr.write(f"Worker: Project Root: {PROJECT_ROOT}\n")
    sys.stderr.write(f"Worker: Config File: {CONFIG_FILE}\n")
    sys.stderr.write(f"Worker: Checkpoint File: {CHECKPOINT_FILE}\n")
    sys.stderr.write(f"Worker: Label Map File: {LABEL_MAP_FILE}\n")

    sys.stderr.write("Worker: Initializing ST-GCN++ model...\n")
    
    # Load the model only once
    if not load_model():
        sys.stderr.write("Worker: Model loading failed. Exiting.\n")
        sys.exit(1)
    
    sys.stderr.write("Worker: Model initialization complete. Waiting for input...\n")
    
    # Main loop to read JSON from stdin
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line: # Empty line means EOF or disconnected pipe
                # If pipe is disconnected, typically the process will be killed by Node.js
                # For robustness, we might want to break or sleep, but for child_process, just continue
                # sys.stderr.write("Worker: Received empty line, possibly pipe closed.\n")
                continue 
            
            input_data = json.loads(line)
            request_type = input_data.get("type", "landmarks")
            
            if request_type == "landmarks":
                # Get the keypoints array from the input
                # With the updated client code, we expect an array of (V, C) format keypoints
                # where V=27 joints and C=3 channels (x, y, visibility)
                current_frame_keypoints = input_data.get("keypoints") 
                
                if not current_frame_keypoints or not isinstance(current_frame_keypoints, list) or len(current_frame_keypoints) == 0:
                    response = {"error": "Invalid input: 'keypoints' for current frame missing or empty."}
                    sys.stdout.write(json.dumps(response) + '\n')
                    sys.stdout.flush()
                    continue
                # Log the received data format for debugging
                sys.stderr.write(f"Worker: Received keypoints shape: {len(current_frame_keypoints)}x{len(current_frame_keypoints[0]) if len(current_frame_keypoints) > 0 else 0}\n")
                
                try:
                    # Process keypoints data
                    # Check if we received a single frame or multiple frames
                    if isinstance(current_frame_keypoints[0][0], list):
                        # Multiple frames: [[frame1], [frame2], ...] where each frame is (V, C)
                        for frame_keypoints in current_frame_keypoints:
                            keypoint_buffer_instance.add_frame(np.array(frame_keypoints, dtype=np.float32))
                    else:
                        # Single frame: direct (V, C) format
                        keypoint_buffer_instance.add_frame(np.array(current_frame_keypoints, dtype=np.float32))
                except Exception as e:
                    sys.stderr.write(f"Worker: Error processing keypoints: {e}\n")
                    sys.stderr.write(traceback.format_exc(file=sys.stderr))
                
                frame_idx_counter += 1

                # Only run inference periodically and if buffer is full enough
                if frame_idx_counter % PREDICT_PER_NFRAME == 0: # and len(keypoint_buffer_instance.buffer) >= NUM_FRAMES:
                    # The run_inference function itself checks buffer length
                    results = run_inference()
                else:
                    # If not running inference, just provide buffer status
                    results = {
                        "prediction": "Buffering",
                        "message": f"Buffering frames... ({len(keypoint_buffer_instance.buffer)}/{NUM_FRAMES})",
                        "label_index": -1,
                        "score": 0.0,
                        "buffer_length": len(keypoint_buffer_instance.buffer),
                        "required_frames": NUM_FRAMES
                    }
                
                sys.stdout.write(json.dumps(results) + '\n')
                sys.stdout.flush()
                
            elif request_type == "reset_buffer": # Added a dedicated reset command
                keypoint_buffer_instance = KeypointBuffer(buffer_size=NUM_FRAMES, step_size=SLIDING_WINDOW_STEP)
                last_predictions_deque.clear()
                frame_idx_counter = 0 # Reset frame counter on buffer reset
                sys.stdout.write(json.dumps({"message": "Buffer and predictions reset", "status": "reset"}) + '\n')
                sys.stdout.flush()
                
            elif request_type == "ping":
                sys.stdout.write(json.dumps({"status": "alive"}) + '\n')
                sys.stdout.flush()
                
            else:
                sys.stderr.write(f"Worker: Unknown request type: {request_type}\n")
                sys.stdout.write(json.dumps({"error": f"Unknown reqKeypointBufferuest type: {request_type}"}) + '\n')
                sys.stdout.flush()
                
        except json.JSONDecodeError:
            sys.stderr.write("Worker: Error: Invalid JSON received. Skipping line.\n")
            sys.stdout.write(json.dumps({"error": "Invalid JSON input to worker."}) + '\n')
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Worker: Critical error in main loop: {e}\n")
            sys.stderr.write(traceback.format_exc(file=sys.stderr))
            sys.stdout.write(json.dumps({"error": f"Worker internal error: {str(e)}"}) + '\n')
            sys.stdout.flush()

if __name__ == "__main__":
    main()