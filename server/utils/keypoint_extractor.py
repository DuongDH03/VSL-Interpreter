import numpy as np
import json
import cv2
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import sys # Import sys for stderr printing

# Constants for ST-GCN++ input format
NUM_FRAMES = 64  # This should be dictated by your model's config (stgcnpp_hand27.py)
NUM_JOINTS = 27  # Correct for hand_body_27 layout
NUM_CHANNELS = 3  # X, Y, confidence/visibility coordinates

# --- MediaPipe Index Constants (for clarity) ---
# Pose landmarks (33 points)
MP_POSE_NOSE = 0
MP_POSE_LEFT_SHOULDER = 11
MP_POSE_RIGHT_SHOULDER = 12
MP_POSE_LEFT_ELBOW = 13
MP_POSE_RIGHT_ELBOW = 14
MP_POSE_LEFT_WRIST = 15
MP_POSE_RIGHT_WRIST = 16
# Additional body joints for hand_body_27 if needed:
MP_POSE_LEFT_EYE = 2
MP_POSE_RIGHT_EYE = 5

# Hand landmarks (21 points per hand)
# Common points used for 10-point hand skeletons from MediaPipe:
# WRIST: 0
# THUMB: CMC=1, MCP=2, IP=3, TIP=4
# INDEX: MCP=5, PIP=6, DIP=7, TIP=8
# MIDDLE: MCP=9, PIP=10, DIP=11, TIP=12
# RING: MCP=13, PIP=14, DIP=15, TIP=16
# PINKY: MCP=17, PIP=18, DIP=19, TIP=20

# We need a specific mapping for 10 hand points per hand
MP_HAND_WRIST = 0 # MediaPipe hand wrist
MP_HAND_THUMB_MCP = 2
MP_HAND_THUMB_TIP = 4
MP_HAND_INDEX_MCP = 5
MP_HAND_INDEX_TIP = 8
MP_HAND_MIDDLE_MCP = 9
MP_HAND_MIDDLE_TIP = 12
MP_HAND_RING_MCP = 13
MP_HAND_RING_TIP = 16
MP_HAND_PINKY_MCP = 17
MP_HAND_PINKY_TIP = 20

# Define the 27-joint target mapping
# This is a hypothesized mapping based on standard hand_body_27 definitions.
STGCN_JOINT_MAP = {
    # Body joints (7 joints from MediaPipe Pose)
    0: {'type': 'pose', 'mp_idx': MP_POSE_NOSE},
    1: {'type': 'pose', 'mp_idx': MP_POSE_LEFT_EYE},
    2: {'type': 'pose', 'mp_idx': MP_POSE_RIGHT_EYE},
    3: {'type': 'pose', 'mp_idx': MP_POSE_LEFT_SHOULDER},
    4: {'type': 'pose', 'mp_idx': MP_POSE_RIGHT_SHOULDER},
    5: {'type': 'pose', 'mp_idx': MP_POSE_LEFT_ELBOW},
    6: {'type': 'pose', 'mp_idx': MP_POSE_RIGHT_ELBOW},
    
    # Left Hand joints (10 joints from MediaPipe Left Hand)
    7: {'type': 'hand', 'mp_idx': MP_HAND_WRIST, 'handedness': 'Right'},
    8: {'type': 'hand', 'mp_idx': MP_HAND_THUMB_MCP, 'handedness': 'Right'},
    9: {'type': 'hand', 'mp_idx': MP_HAND_THUMB_TIP, 'handedness': 'Right'},
    10: {'type': 'hand', 'mp_idx': MP_HAND_INDEX_MCP, 'handedness': 'Right'},
    11: {'type': 'hand', 'mp_idx': MP_HAND_INDEX_TIP, 'handedness': 'Right'},
    12: {'type': 'hand', 'mp_idx': MP_HAND_MIDDLE_MCP, 'handedness': 'Right'},
    13: {'type': 'hand', 'mp_idx': MP_HAND_MIDDLE_TIP, 'handedness': 'Right'},
    14: {'type': 'hand', 'mp_idx': MP_HAND_RING_MCP, 'handedness': 'Right'},
    15: {'type': 'hand', 'mp_idx': MP_HAND_RING_TIP, 'handedness': 'Right'},
    16: {'type': 'hand', 'mp_idx': MP_HAND_PINKY_TIP, 'handedness': 'Right'},

    # Right Hand joints (10 joints from MediaPipe Right Hand)
    17: {'type': 'hand', 'mp_idx': MP_HAND_WRIST, 'handedness': 'Left'},
    18: {'type': 'hand', 'mp_idx': MP_HAND_THUMB_MCP, 'handedness': 'Left'},
    19: {'type': 'hand', 'mp_idx': MP_HAND_THUMB_TIP, 'handedness': 'Left'},
    20: {'type': 'hand', 'mp_idx': MP_HAND_INDEX_MCP, 'handedness': 'Left'},
    21: {'type': 'hand', 'mp_idx': MP_HAND_INDEX_TIP, 'handedness': 'Left'},
    22: {'type': 'hand', 'mp_idx': MP_HAND_MIDDLE_MCP, 'handedness': 'Left'},
    23: {'type': 'hand', 'mp_idx': MP_HAND_MIDDLE_TIP, 'handedness': 'Left'},
    24: {'type': 'hand', 'mp_idx': MP_HAND_RING_MCP, 'handedness': 'Left'},
    25: {'type': 'hand', 'mp_idx': MP_HAND_RING_TIP, 'handedness': 'Left'},
    26: {'type': 'hand', 'mp_idx': MP_HAND_PINKY_TIP, 'handedness': 'Left'},
}

def extract_keypoints(
    pose_landmarks: Optional[Any] = None, 
    hand_landmarks: Optional[Any] = None,
    handedness_list: Optional[Any] = None
) -> np.ndarray:
    """
    Extract keypoints from MediaPipe pose and hand landmarks and format for ST-GCN++.
    This function must produce a single frame's keypoints in (NUM_JOINTS, NUM_CHANNELS) shape.
    
    Args:
        pose_landmarks: MediaPipe PoseLandmarks object (results.pose_landmarks)
        hand_landmarks: List of MediaPipe HandLandmarks objects (results.multi_hand_landmarks)
        handedness_list: List of MediaPipe Handedness objects (results.multi_handedness)
    Returns:
        np.ndarray: Formatted keypoints for ST-GCN++ with shape (NUM_JOINTS, NUM_CHANNELS)
                    where NUM_JOINTS=27, NUM_CHANNELS=3 (x,y,confidence)
    """
    combined_keypoints = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    
    # Populate body joints from pose_landmarks
    if pose_landmarks:
        for stgcn_idx in range(7): # Assuming first 7 joints are body
            map_info = STGCN_JOINT_MAP.get(stgcn_idx)
            if map_info and map_info['type'] == 'pose':
                mp_idx = map_info['mp_idx']
                if mp_idx < len(pose_landmarks.landmark):
                    lm = pose_landmarks.landmark[mp_idx]
                    combined_keypoints[stgcn_idx] = [lm.x, lm.y, lm.visibility]

    # Populate hand joints from hand_landmarks
    if hand_landmarks and handedness_list:
        for i, (h_landmarks, h_handedness) in enumerate(zip(hand_landmarks, handedness_list)):
            hand_label = h_handedness.classification[0].label # 'Left' or 'Right'
            
            # MediaPipe's 'Left' hand is usually the one on the *right* side of the image (viewer's perspective)
            # MediaPipe's 'Right' hand is usually the one on the *left* side of the image (viewer's perspective)

            # Iterate through the STGCN_JOINT_MAP for hand joints
            for stgcn_idx in range(7, NUM_JOINTS): # Start from joint 7 for hands
                map_info = STGCN_JOINT_MAP.get(stgcn_idx)
                if map_info and map_info['type'] == 'hand' and map_info['handedness'] == hand_label:
                    mp_idx = map_info['mp_idx']
                    if mp_idx < len(h_landmarks.landmark):
                        lm = h_landmarks.landmark[mp_idx]
                        combined_keypoints[stgcn_idx] = [lm.x, lm.y, 1.0] # MediaPipe hands don't have visibility, use 1.0

    return combined_keypoints

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to be invariant to camera position and person size.
    Args:
        keypoints: Array with shape (T, V, C) - Note: KeypointBuffer.get_input() produces this.
        
    Returns:
        np.ndarray: Normalized keypoints
    """
    if keypoints.ndim != 3 or keypoints.shape[2] != NUM_CHANNELS:
        sys.stderr.write(f"Warning: normalize_keypoints received unexpected shape: {keypoints.shape}. Skipping normalization.\n")
        return keypoints

    # Center keypoints around the nose joint (joint 0 in your 27-joint layout)
    root_joint_index = 0 
    
    # Calculate root position for each frame: shape (T, 1, C)
    root = keypoints[:, root_joint_index:root_joint_index+1, :] 
    
    # Center all keypoints
    centered_keypoints = keypoints - root
    
    # Scale normalization
    # Calculate scale for each frame independently using x,y coordinates
    # shape: (T,)
    scale = np.linalg.norm(centered_keypoints[:, :, :2], axis=(1, 2), keepdims=True) + 1e-8
    
    normalized_keypoints = centered_keypoints / scale 

    # For the confidence channel, keep original confidence values
    normalized_keypoints[:, :, 2] = keypoints[:, :, 2] 

    return normalized_keypoints

class KeypointBuffer:
    """Buffer to collect frames for fixed-length input to ST-GCN++."""
    
    def __init__(self, buffer_size: int, step_size: int):
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.buffer = deque(maxlen=buffer_size * 2) # Using deque for efficient appends/pops
        sys.stderr.write(f"KeypointBuffer initialized with buffer_size={buffer_size}, step_size={step_size}\n")
        
    def add_frame(self, keypoints: np.ndarray) -> None:
        """
        Add keypoints from a new frame to the buffer.
        keypoints is expected to be a NumPy array of shape (V, C), e.g., (27, 3).
        """
        # CRITICAL VALIDATION: Ensure incoming keypoints match expected NUM_JOINTS
        if keypoints.shape != (NUM_JOINTS, NUM_CHANNELS):
            sys.stderr.write(f"KeypointBuffer: WARNING: Expected (V,C) input ({NUM_JOINTS},{NUM_CHANNELS}) but got shape {keypoints.shape}. Attempting to pad/truncate.\n")
            # Create a correctly shaped array and fill with existing data
            padded_keypoints = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
            min_V = min(keypoints.shape[0], NUM_JOINTS)
            min_C = min(keypoints.shape[1] if keypoints.ndim > 1 else 0, NUM_CHANNELS)
            
            if min_V > 0 and min_C > 0:
                padded_keypoints[:min_V, :min_C] = keypoints[:min_V, :min_C]
            keypoints = padded_keypoints
        
        self.buffer.append(keypoints)
        
    def get_input(self) -> np.ndarray:
        """
        Get stacked keypoints as input for ST-GCN++.
        Pads with repeated first frame if buffer is not full.
        Returns shape (T, V, C).
        """
        if len(self.buffer) == 0:
            sys.stderr.write("KeypointBuffer: Buffer is empty, returning zeros.\n")
            return np.zeros((self.buffer_size, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
        
        frames_to_stack = list(self.buffer) # Convert deque to list for easier slicing

        if len(frames_to_stack) < self.buffer_size:
            # If buffer is not full, pad with the first available frame to reach buffer_size
            padding_needed = self.buffer_size - len(frames_to_stack)
            # Use `first_frame` for padding only if `frames_to_stack` is not empty
            first_frame = frames_to_stack[0].copy() if frames_to_stack else np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
            padding = [first_frame for _ in range(padding_needed)]
            frames_to_stack = padding + frames_to_stack
        else:
            # Use the most recent `buffer_size` frames from the end of the deque
            frames_to_stack = frames_to_stack[-self.buffer_size:]
            
        # Stack all frames along first dimension to get (T, V, C)
        stacked = np.stack(frames_to_stack, axis=0)
        
        # Apply normalization here, as it expects (T, V, C)
        return normalize_keypoints(stacked)
    
    def slide_window(self) -> None:
        """Slide window by step_size frames."""
        for _ in range(self.step_size):
            if len(self.buffer) > 0:
                self.buffer.popleft() # Efficiently remove from left

    def reset(self) -> None:
        """Clears the buffer."""
        self.buffer.clear()
        sys.stderr.write("KeypointBuffer: Buffer reset.\n")
