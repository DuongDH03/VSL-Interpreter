import numpy as np
import json
import cv2
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple

# Constants for ST-GCN++ input format
NUM_FRAMES = 100  # ST-GCN++ expects a fixed number of frames
NUM_JOINTS = 27  # Combined joint count (21 per hand * 2 + 33 pose keypoints - duplicates)
NUM_CHANNELS = 3  # X, Y, confidence coordinates

def extract_keypoints(
    hand_landmarks: Optional[List[Dict[str, Any]]] = None,
    pose_landmarks: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Extract keypoints from MediaPipe hand and pose landmarks and format for ST-GCN++.
    
    Args:
        hand_landmarks: List of hand landmarks from MediaPipe
        pose_landmarks: List of pose landmarks from MediaPipe
          Returns:
        np.ndarray: Formatted keypoints for ST-GCN++ with shape (C, T, V)
        where C=3 (x,y,confidence), T=frames, V=joints
    """
    # Initialize empty array for keypoints
    keypoints = np.zeros((NUM_CHANNELS, 1, NUM_JOINTS))
    
    # Process hand landmarks if available
    if hand_landmarks:
        # MediaPipe returns landmarks for each hand separately
        hand_idx = 0
        for hand in hand_landmarks:
            # Each hand has 21 landmarks
            for i, landmark in enumerate(hand):                # MediaPipe landmark format is {x, y, z}
                # Convert to array format [x, y, confidence]
                # Use 1.0 as default confidence for hand landmarks
                keypoints[:, 0, hand_idx * 21 + i] = [landmark.get('x', 0), 
                                                    landmark.get('y', 0), 
                                                    1.0]  # Fixed confidence instead of z-coordinate
            hand_idx += 1
            if hand_idx >= 2:
                break  # Only process up to 2 hands
    
    # Process pose landmarks if available
    if pose_landmarks:
        # MediaPipe pose has 33 landmarks
        # We'll store them after the hand landmarks
        for i, landmark in enumerate(pose_landmarks):
            # Skip landmarks that may conflict with hands
            # Also store the key pose points that are relevant for sign language            if i < 33:  # Total pose landmarks
                keypoints[:, 0, 42 + i] = [landmark.get('x', 0), 
                                          landmark.get('y', 0), 
                                          landmark.get('visibility', 1.0)]  # Use visibility as confidence
    
    return keypoints

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to be invariant to camera position and person size.
    
    Args:
        keypoints: Array with shape (C, T, V)
        
    Returns:
        np.ndarray: Normalized keypoints
    """
    # Center keypoints around the root (mid hip)
    # For sign language, we might use a different reference point
    root = keypoints[:, :, 0:1]  # Use the first joint as reference
    keypoints = keypoints - root
    
    # Scale normalization
    scale = np.sqrt(np.sum(np.square(keypoints)))
    return keypoints / (scale + 1e-8)  # Adding epsilon to avoid division by zero

class KeypointBuffer:
    """Buffer to collect frames for fixed-length input to ST-GCN++."""
    
    def __init__(self, buffer_size: int = NUM_FRAMES, step_size: int = 8):
        """
        Initialize keypoint buffer.
        
        Args:
            buffer_size: Number of frames to collect
            step_size: Number of frames to slide the window by
        """
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.buffer = []
        
    def add_frame(self, keypoints: np.ndarray) -> bool:
        """
        Add keypoints from a new frame to the buffer.
        
        Args:
            keypoints: Array with shape (V, C)
            
        Returns:
            bool: True if buffer is full and ready for processing
        """
        # Handle different possible input shapes
        if keypoints.ndim == 3 and keypoints.shape[0] == 3 and keypoints.shape[1] == 1:
            # Input is (C, T=1, V) format from old extract_keypoints
            keypoints = np.transpose(keypoints[:, 0, :], (1, 0))  # Convert to (V, C)
            
        # Ensure keypoints are in (V, C) format
        if keypoints.ndim != 2:
            print(f"KeypointBuffer: Warning - expected (V,C) input but got shape {keypoints.shape}\n")
            
        # Add to buffer
        self.buffer.append(keypoints)
        
        # If buffer exceeds size, remove oldest frames
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            
        # Return True if buffer is full and ready for processing
        return len(self.buffer) >= self.buffer_size

        
    def get_input(self) -> np.ndarray:
        """
        Get stacked keypoints as input for ST-GCN++.
        
        Returns:
            np.ndarray: Array with shape (T, V, C)
        """
        # If buffer is not full, pad with zeros or first frame
        if len(self.buffer) == 0:
            # Empty buffer, return zeros
            return np.zeros((self.buffer_size, NUM_JOINTS, NUM_CHANNELS))
        
        if len(self.buffer) < self.buffer_size:
            # Pad with repeated first frame
            pad_size = self.buffer_size - len(self.buffer)
            padding = [self.buffer[0].copy() for _ in range(pad_size)]  # Use first frame
            frames_to_stack = padding + self.buffer
        else:
            # Use the most recent buffer_size frames
            frames_to_stack = self.buffer[-self.buffer_size:]
        
        # Stack frames along first dimension to get (T, V, C)
        stacked = np.stack(frames_to_stack, axis=0)
        return stacked
    
    def slide_window(self) -> None:
        """Slide window by step_size frames."""
        if len(self.buffer) >= self.step_size:
            self.buffer = self.buffer[self.step_size:]

    def reset(self):
        """Clears the buffer."""
        self.buffer.clear()

def process_frame(frame: np.ndarray, mp_hands, mp_pose) -> Tuple[List, List]:
    """
    Process a video frame with MediaPipe to extract hand and pose landmarks.
    
    Args:
        frame: Input video frame
        mp_hands: MediaPipe Hands model instance
        mp_pose: MediaPipe Pose model instance
        
    Returns:
        Tuple containing hand landmarks and pose landmarks
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    hand_results = mp_hands.process(frame_rgb)
    pose_results = mp_pose.process(frame_rgb)
    
    # Extract hand landmarks
    hand_landmarks_list = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:            landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'visibility': 1.0  # Use fixed confidence for hands as they don't have visibility
                })
            hand_landmarks_list.append(landmarks)
    
    # Extract pose landmarks
    pose_landmarks_list = []
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:            pose_landmarks_list.append({
                'x': lm.x,
                'y': lm.y,
                'visibility': lm.visibility  # Use actual visibility for pose landmarks
            })
    
    return hand_landmarks_list, pose_landmarks_list

def process_video(video_path: str, buffer_size: int = NUM_FRAMES, step_size: int = 8) -> List[np.ndarray]:
    """
    Process a video file for sign language recognition.
    
    Args:
        video_path: Path to video file
        buffer_size: Number of frames in each sequence
        step_size: Number of frames to slide the window by
        
    Returns:
        List[np.ndarray]: List of keypoint sequences for ST-GCN++ input
    """
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
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    buffer = KeypointBuffer(buffer_size, step_size)
    sequences = []
    
    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame with MediaPipe
        hand_landmarks, pose_landmarks = process_frame(frame, mp_hands, mp_pose)
        
        # Extract keypoints
        keypoints = extract_keypoints(hand_landmarks, pose_landmarks)
        
        # Add to buffer
        is_ready = buffer.add_frame(keypoints)
        
        # If buffer is ready, add sequence to list
        if is_ready and len(buffer.buffer) == buffer_size:
            input_seq = buffer.get_input()
            sequences.append(input_seq)
            buffer.slide_window()
    
    # Clean up
    cap.release()
    mp_hands.close()
    mp_pose.close()
    
    return sequences

def save_to_file(sequences: List[np.ndarray], output_file: str) -> None:
    """
    Save sequences to file for ST-GCN++ input.
    
    Args:
        sequences: List of keypoint sequences
        output_file: Path to output file
    """
    # Convert to list for JSON serialization
    json_data = [seq.tolist() for seq in sequences]
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_data, f)

def process_landmarks_for_stgcn(hand_landmarks: List, pose_landmarks: List) -> np.ndarray:
    """
    Process incoming landmarks from client into ST-GCN++ format.
    
    Args:
        hand_landmarks: Hand landmarks from client
        pose_landmarks: Pose landmarks from client
        
    Returns:
        np.ndarray: Keypoints formatted for ST-GCN++
    """
    keypoints = extract_keypoints(hand_landmarks, pose_landmarks)
    return keypoints
