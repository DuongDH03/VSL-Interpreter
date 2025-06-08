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
    # The frontend extracts (V, C) per frame. My `extractKeypointsJs` produced `(1, V, C)` as the `currentUnifiedKeypoints`
    # and the `onFrameResults` appends `currentUnifiedKeypoints` (which is `[[[x,y,v], ...]]`) to `keypointsBuffer`.
    # So `keypoint_buffer_instance.get_input()` from the client would return a list of these,
    # and would need to be stacked.
    # Let's assume keypoint_buffer_instance.get_input() now returns the stacked (T, V, C) NumPy array
    
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
        data_processed_by_pipeline = test_pipeline_instance(fake_anno)
        # The pipeline should output a dict with 'keypoint' tensor in (C, T, V, M) format
        input_tensor_for_model = data_processed_by_pipeline['keypoint']
    else: # For placeholder model, manual tensor creation
        # Placeholder expects (N, C, T, V, M)
        # Convert (1, T, V, C) to (1, C, T, V, 1) or (N, C, T, V, M)
        # If input_for_pipeline is (1, T, V, C)
        input_tensor_for_model = torch.tensor(input_for_pipeline, dtype=torch.float32).permute(0, 3, 1, 2) # -> (1, C, T, V)
        input_tensor_for_model = input_tensor_for_model.unsqueeze(-1) # -> (1, C, T, V, 1) M=1
    
    device = next(recognizer.parameters()).device
    input_tensor_for_model = input_tensor_for_model.to(device)

    # Forward pass
    with torch.no_grad():
        output_logits = recognizer(input_tensor_for_model, return_loss=False)[0] # Assuming batch size 1, get first element
        
        # Get probabilities with softmax
        probabilities = F.softmax(output_logits, dim=0).cpu().numpy() # Output is typically 1D array of probabilities

    # Get top-k predictions
    top_k = 3 
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    
    class_names = get_class_names()
    top_predictions = [class_names.get(str(idx), f"unknown_{idx}") for idx in top_k_indices]
    
    # Update last predictions for smoothing
    if top_k_probs[0] > 0.3: # Using a confidence threshold from your original demo
        last_predictions_deque.append(top_predictions[0])
    
    # Apply temporal smoothing
    smoothed_prediction = "unknown"
    smoothed_confidence = 0.0
    if last_predictions_deque:
        from collections import Counter
        prediction_counts = Counter(last_predictions_deque)
        most_common_entry = prediction_counts.most_common(1)[0]
        smoothed_prediction = most_common_entry[0]
        smoothed_confidence = most_common_entry[1] / len(last_predictions_deque)
        
    return {
        "message": "Inference complete",
        "raw_top_predictions": top_predictions,
        "raw_confidences": top_k_probs.tolist(),
        "prediction": smoothed_prediction,
        "score": float(smoothed_confidence),
        "label_index": int(top_k_indices[0]) # Return index of top raw prediction
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
                # The frontend sends `unifiedKeypoints` (single frame, (1, V, C) or (V, C) format)
                # It's an array of arrays from JS: `[[x,y,v], [x,y,v], ...]` (for one person's frame)
                current_frame_keypoints = input_data.get("keypoints") 
                
                if not current_frame_keypoints or not isinstance(current_frame_keypoints, list) or len(current_frame_keypoints) == 0:
                    response = {"error": "Invalid input: 'keypoints' for current frame missing or empty."}
                    sys.stdout.write(json.dumps(response) + '\n')
                    sys.stdout.flush()
                    continue

                # Add the current frame's keypoints to the buffer
                # KeypointBuffer's add_frame should expect a (V, C) numpy array or compatible list of lists
                # The frontend's `extractKeypointsJs` returns `[[[x,y,v],...]]` so it's a list containing one `(V,C)` array.
                # We need to pass the inner (V,C) array to `add_frame`.
                # If frontend sends `keypoints: unifiedKeypoints[0]` then it's directly `(V,C)`
                # Let's assume frontend sends `keypoints: unifiedKeypoints[0]` (the V,C array)
                # (This is what `App.jsx`'s `sendLandmarksToAPI(landmarksData.unifiedKeypoints)` will do if `landmarksData.unifiedKeypoints` is `[[[x,y,v],...]]` and you send `unifiedKeypoints[0]`)
                keypoint_buffer_instance.add_frame(np.array(current_frame_keypoints, dtype=np.float32))
                
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
                sys.stdout.write(json.dumps({"error": f"Unknown request type: {request_type}"}) + '\n')
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