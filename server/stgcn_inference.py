import sys
import json
import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# Check if PyTorch is available
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"Warning: PyTorch could not be loaded: {e}")
    print("Using placeholder inference without machine learning model.")
from utils.keypoint_extractor import (
    extract_keypoints,
    normalize_keypoints, 
    KeypointBuffer,
    process_frame
)

    # Check if config exists, otherwise use default values
try:
    # Add the directory containing the script to the path to make imports work correctly
    server_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(server_dir)
    
    # Try to import model configuration from our local models directory
    # Read model config from stgcnpp_hand27.py
    CONFIG_PATH = os.path.join(server_dir, 'models', 'configs', 'stgcnpp_hand27.py')
    if os.path.exists(CONFIG_PATH):
        # Read the config file and extract values
        with open(CONFIG_PATH, 'r') as f:
            config_content = f.read()
            
        # Extract window size from config (default to 64 if not found)
        import re
        window_size_match = re.search(r'clip_len=(\d+)', config_content)
        NUM_FRAMES = int(window_size_match.group(1)) if window_size_match else 64
        SLIDING_WINDOW_STEP = 8  # Default step size
        
        print(f"Config loaded from {CONFIG_PATH}. Window size: {NUM_FRAMES}")
    else:
        raise ImportError(f"Config file not found at {CONFIG_PATH}")
    
    # Check for class mapping
    CLASS_MAP_PATH = os.path.join(server_dir, 'models', 'class_map.json')
    if os.path.exists(CLASS_MAP_PATH):
        import json
        with open(CLASS_MAP_PATH, 'r') as f:
            CLASS_MAP = json.load(f)
            print(f"Loaded class mapping with {len(CLASS_MAP)} classes")
    else:
        CLASS_MAP = {str(i): f"sign_{i}" for i in range(10)}
        print("Using default class mapping")
    
    def get_class_names():
        return CLASS_MAP
    
    # Import model - assuming the model class is properly defined elsewhere
    # Not importing directly since we don't have PYSKL's structure
        
except Exception as e:
    # Default values if config not found
    print(f"Warning: Failed to load config: {e}")
    print("Using default values")
    NUM_FRAMES = 64
    SLIDING_WINDOW_STEP = 8
    
    # Define a basic get_class_names function
    def get_class_names():
        return {str(i): f"sign_{i}" for i in range(10)}

    # Global variables for prediction smoothing
last_predictions = deque(maxlen=5)  # Default smoothing window size

def load_model():
    """
    Load the ST-GCN++ model.
    
    Returns:
        model: The ST-GCN++ model
    """
    try:
        # Try to import STGCN_PlusPlus or create a placeholder class
        try:
            from pyskl.models.gcn import STGCN_PlusPlus
        except ImportError:
            print("Warning: Failed to import STGCN_PlusPlus from PYSKL.")
            print("Trying simplified model import...")
            
            # Create a simple placeholder class for development/testing
            class STGCN_PlusPlus:
                def __init__(self, **kwargs):
                    self.params = kwargs
                
                def to(self, device):
                    return self
                
                def eval(self):
                    return self
                    
                def __call__(self, x):
                    import torch
                    # Return random predictions for testing
                    return torch.rand(x.shape[0], 10)
                    
                def parameters(self):
                    import torch
                    yield torch.nn.Parameter(torch.zeros(1))
        
        # Define model config
        graph_args = {
            'layout': 'hand_body_27',
            'strategy': 'spatial'
        }
        
        # Create model - ensure we're set up for x, y, confidence format
        model = STGCN_PlusPlus(
            in_channels=3,  # 3 channels: x, y, confidence
            hidden_channels=64,
            hidden_dim=256,
            num_class=10,
            graph_args=graph_args,
            dropout=0.2
        )
        
        # Try to load model weights
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'stgcn++_model.pth')
        if os.path.exists(model_path):
            # Load the model with explicit CPU map_location to avoid CUDA errors
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Model loaded from {model_path}")
        else:
            print("Model weights not found. Using random initialization for development.")
        
        # Set model to evaluation mode
        model.eval()
        
        # Use CPU explicitly instead of CUDA to avoid memory errors
        device = torch.device("cpu")
        model = model.to(device)
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_inference(model, keypoint_sequences):
    """
    Run inference on keypoint sequences.
    
    Args:
        model: ST-GCN++ model
        keypoint_sequences: List of keypoint sequences in format (C,T,V,M)
        
    Returns:
        dict: Results of inference with predictions and confidences
    """
    if model is None or not TORCH_AVAILABLE:
        return {
            "message": "Model not loaded or PyTorch not available",
            "top_predictions": ["placeholder"],
            "confidences": [1.0]
        }
    
    if not keypoint_sequences:
        return {
            "message": "No valid sequences to process",
            "top_predictions": [],
            "confidences": []
        }
    
    # Get class names
    try:
        class_names = get_class_names()
    except:
        class_names = {str(i): f"sign_{i}" for i in range(10)}
    
    # Process each sequence
    results = []
    
    for sequence in keypoint_sequences:
        try:            # Prepare input tensor
            # Convert from (C,T,V) to (N,C,T,V,M) format expected by the model
            # where N=batch size (1), C=channels (3), T=frames, V=joints, M=persons (1)
            device = next(model.parameters()).device
            
            # Ensure correct format for PYSKL
            # PYSKL expects channels in order [x, y, confidence]
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            # Add batch dimension (N) and person dimension (M)
            x = sequence_tensor.unsqueeze(0).unsqueeze(-1).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
                
                # Get probabilities with softmax
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                
                # Get top-k predictions
                top_k = 3  # Get top 3 predictions
                top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
                top_k_probs = probabilities[top_k_indices]
                
                # Convert indices to class names
                top_predictions = [class_names.get(str(idx), f"unknown_{idx}") for idx in top_k_indices]
                
                # Add to results
                results.append({
                    "top_predictions": top_predictions,
                    "confidences": top_k_probs.tolist()
                })
                
                # Update last predictions for smoothing if confidence is high enough
                if top_k_probs[0] > 0.6:  # Confidence threshold
                    last_predictions.append(top_predictions[0])
        except Exception as e:
            print(f"Error during inference: {e}")
    
    # Apply temporal smoothing
    if last_predictions:
        # Count occurrences of each prediction
        from collections import Counter
        prediction_counts = Counter(last_predictions)
        
        # Get most common prediction
        most_common = prediction_counts.most_common(1)[0][0]
        confidence = prediction_counts.most_common(1)[0][1] / len(last_predictions)
    else:
        most_common = "unknown"
        confidence = 0.0
    
    return {
        "message": "Inference complete",
        "num_sequences": len(keypoint_sequences),
        "results": results,
        "smoothed_prediction": most_common,
        "smoothed_confidence": confidence
    }

def process_live_landmarks(hand_landmarks, pose_landmarks):
    """
    Process live landmarks from client.
    
    Args:
        hand_landmarks: Hand landmarks from client
        pose_landmarks: Pose landmarks from client
        
    Returns:
        dict: Processed landmarks or inference result
    """
    try:
        # Extract keypoints in the format needed for the model
        keypoints = extract_keypoints(hand_landmarks, pose_landmarks)
        
        # Initialize keypoint buffer
        buffer = KeypointBuffer(buffer_size=NUM_FRAMES, step_size=SLIDING_WINDOW_STEP)
        
        # Add frame to buffer
        buffer.add_frame(keypoints)
        
        # If we have a model, run inference
        model = load_model()
        if model:
            # Get input sequence from buffer
            # For live inference, we'll use whatever we have
            input_seq = buffer.get_input()
            
            # Run inference with the current buffer
            results = run_inference(model, [input_seq])
            
            # Add the keypoints to the result
            results["keypoints"] = keypoints.tolist()
            return results
        
        # If no model or still building buffer, just return the keypoints
        return {
            "keypoints": keypoints.tolist(),
            "message": "Landmarks processed successfully"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error processing landmarks: {str(e)}"
        }

def process_video_with_mediapipe(video_path):
    """
    Process video with MediaPipe and extract keypoint sequences.
    
    Args:
        video_path: Path to video file
        
    Returns:
        list: List of keypoint sequences
    """
    print(f"Processing video at {video_path}...")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize keypoint buffer
    buffer = KeypointBuffer(buffer_size=NUM_FRAMES, step_size=SLIDING_WINDOW_STEP)
    keypoint_sequences = []
    
    try:
        frame_count = 0
        while cap.isOpened():
            # Read frame
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame with MediaPipe
            hand_landmarks_list, pose_landmarks_list = process_frame(frame, mp_hands, mp_pose)
            
            # Extract keypoints
            keypoints = extract_keypoints(hand_landmarks_list, pose_landmarks_list)
            
            # Add frame to buffer
            is_buffer_ready = buffer.add_frame(keypoints)
            
            # If buffer is ready, extract sequence and slide window
            if is_buffer_ready:
                keypoint_sequence = buffer.get_input()
                keypoint_sequences.append(keypoint_sequence)
                buffer.slide_window()
            
            frame_count += 1
            if frame_count % 30 == 0:  # Log progress every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Clean up
        cap.release()
        mp_hands.close()
        mp_pose.close()
    
    print(f"Extracted {len(keypoint_sequences)} sequences from video")
    return keypoint_sequences

def process_image_with_mediapipe(image_path):
    """
    Process image with MediaPipe and extract keypoints.
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Dictionary containing keypoints
    """
    # Initialize MediaPipe hands and pose
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    hand_results = mp_hands.process(image_rgb)
    pose_results = mp_pose.process(image_rgb)
    
    # Extract hand landmarks
    hand_landmarks_list = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:            landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'visibility': 1.0  # Use fixed confidence for hands
                })
            hand_landmarks_list.append(landmarks)
    
    # Extract pose landmarks
    pose_landmarks_list = []
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:            pose_landmarks_list.append({
                'x': lm.x,
                'y': lm.y,
                'visibility': lm.visibility  # Use actual visibility as confidence
            })
    
    # Extract keypoints
    keypoints = extract_keypoints(hand_landmarks_list, pose_landmarks_list)
    
    # Clean up
    mp_hands.close()
    mp_pose.close()
    
    return {
        "keypoints": keypoints.tolist(),
        "hand_landmarks": hand_landmarks_list,
        "pose_landmarks": pose_landmarks_list
    }

def main():
    try:
        # Parse input data from Node.js
        input_data = json.loads(sys.argv[1])
        
        # Get the input type
        input_type = input_data.get("input_type", "video")
        
        if input_type == "video":
            video_path = input_data.get("video_path")
            if not video_path:
                print(json.dumps({"error": "No video path provided"}))
                return
                
            # Process video
            sequences = process_video_with_mediapipe(video_path)
            
            # Load model and run inference
            model = load_model()
            result = run_inference(model, sequences)
            
            print(json.dumps(result))
            
        elif input_type == "image":
            image_path = input_data.get("image_path")
            if not image_path:
                print(json.dumps({"error": "No image path provided"}))
                return
                
            # Process image
            result = process_image_with_mediapipe(image_path)
            print(json.dumps(result))
            
        elif input_type == "live":
            # Process live landmarks
            raw_data = input_data.get("raw_data", {})
            hand_landmarks = raw_data.get("hand_landmarks", [])
            pose_landmarks = raw_data.get("pose_landmarks", [])
            
            # Also check for direct landmark assignment
            if not hand_landmarks:
                hand_landmarks = input_data.get("hand_landmarks", [])
            if not pose_landmarks:
                pose_landmarks = input_data.get("pose_landmarks", [])
            
            # Check if we have any landmarks to process
            if not hand_landmarks and not pose_landmarks:
                print(json.dumps({"error": "No landmarks provided"}))
                return
                  # Direct keypoints provided for inference
            keypoints = input_data.get("keypoints")
            if keypoints:
                # Load model for inference
                model = load_model()
                
                # Convert to numpy array and log shape
                keypoints_array = np.array(keypoints)
                print(f"Debug: Received keypoints with shape {keypoints_array.shape}", file=sys.stderr)
                
                # Check if we need to reshape/restructure the data
                if len(keypoints_array.shape) == 3 and keypoints_array.shape[0] == 3:
                    # Likely already in (C, T, V) format, perfect
                    pass
                elif len(keypoints_array.shape) == 3 and keypoints_array.shape[2] == 3:
                    # Likely in format (T, V, C), needs transpose
                    keypoints_array = np.transpose(keypoints_array, (2, 0, 1))
                    print(f"Debug: Transposed keypoints to shape {keypoints_array.shape}", file=sys.stderr)
                
                # Run inference
                result = run_inference(model, [keypoints_array])
                print(json.dumps(result))
                return
                
            # Process landmarks if no direct keypoints provided
            result = process_live_landmarks(hand_landmarks, pose_landmarks)
            print(json.dumps(result))
            
        else:
            print(json.dumps({"error": f"Unknown input type: {input_type}"}))
    
    except Exception as e:
        # Catch any unexpected errors
        print(json.dumps({"error": f"Error processing request: {str(e)}"}))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
