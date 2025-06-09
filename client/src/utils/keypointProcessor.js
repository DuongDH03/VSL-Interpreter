/**
 * Utility functions for processing MediaPipe hand and pose landmark data
 * Based on the ST-GCN++ demo script for sign language recognition
 */

/**
 * Extracts and formats keypoints from MediaPipe pose and hand results
 * following the same format as used in the ST-GCN++ model
 * 
 * @param {Object} poseData - MediaPipe pose landmark results (poseLandmarks array)
 * @param {Array} handsData - MediaPipe hands landmark results (multiHandLandmarks array)
 * @param {String} layout - Layout to use ('hand_body_27' or 'hand_body_29')
 * @returns {Object} Formatted keypoints ready for model consumption
 */
export function extractKeypoints(poseData, handsData) {
  const numNodes = 27; // Using hand_body_27 exclusively
  
  // Initialize keypoints array with zeros (V=nodes, C=3 coords)
  // Create the direct (V, C) format expected by the Python worker
  const keypoints = Array(numNodes).fill().map(() => [0, 0, 0]);
  
  // Process pose landmarks if available
  if (poseData && poseData.length) {
    const landmarks = poseData;
    
    // 7 body keypoints in hand_body_27: NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW
    const bodyIndices = [0, 2, 5, 11, 12, 13, 14];
    
    bodyIndices.forEach((mpIdx, i) => {
      if (landmarks[mpIdx]) {
        keypoints[i][0] = landmarks[mpIdx].x;
        keypoints[i][1] = landmarks[mpIdx].y;
        keypoints[i][2] = landmarks[mpIdx].visibility || 1.0;
      }
    });
    
    // Left wrist - index 7
    if (landmarks[15]) {
      keypoints[7][0] = landmarks[15].x;
      keypoints[7][1] = landmarks[15].y;
      keypoints[7][2] = landmarks[15].visibility || 1.0;
    }
    
    // Right wrist - index 17
    if (landmarks[16]) {
      keypoints[17][0] = landmarks[16].x;
      keypoints[17][1] = landmarks[16].y;
      keypoints[17][2] = landmarks[16].visibility || 1.0;
    }
  }
  // Process hand landmarks if available
  if (handsData && handsData.length) {
    // In MediaPipe, handedness is relative to the image (mirrored)
    // So "Left" hand in MediaPipe is actually Right hand in real-world
    handsData.forEach(handLandmarks => {      
      // Determine if it's left or right hand
      // In our case we have to infer from position since we don't have handedness data
      // A simple heuristic: if hand is on the left side of image, it's right hand in real-world
      const isLeftHand = determineHandedness(handLandmarks);
      
      // Specific MediaPipe hand landmarks to use:
      // THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, 
      // RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP
      const fingerIndices = [4, 5, 8, 9, 12, 13, 16, 18, 20];
      
      if (isLeftHand) {
        // Left hand in real world (MediaPipe "Right")
        fingerIndices.forEach((mpIdx, i) => {
          keypoints[8 + i][0] = handLandmarks[mpIdx].x;
          keypoints[8 + i][1] = handLandmarks[mpIdx].y;
          keypoints[8 + i][2] = 1.0; // MediaPipe hands don't have visibility
        });
      } else {
        // Right hand in real world (MediaPipe "Left")
        fingerIndices.forEach((mpIdx, i) => {
          keypoints[18 + i][0] = handLandmarks[mpIdx].x;
          keypoints[18 + i][1] = handLandmarks[mpIdx].y;
          keypoints[18 + i][2] = 1.0;
        });
      }
    });
  }
  return {
    keypoint: keypoints,
    layout: 'hand_body_27'
  };
}

/**
 * Simple heuristic to determine handedness without MediaPipe's handedness info
 * Based on position of wrist relative to center of frame
 * 
 * @param {Array} handLandmarks - Single hand landmarks from MediaPipe
 * @returns {Boolean} - True if it's likely left hand, false if right
 */
function determineHandedness(handLandmarks) {
  // Use wrist position (landmark 0) to determine handedness
  const wrist = handLandmarks[0];
  // If wrist x-coordinate is less than 0.5 (left side of image), 
  // it's likely the right hand in the mirror
  return wrist.x < 0.5;
}

/**
 * Prepares a frame of data for the ST-GCN++ model
 * 
 * @param {Object} keypoints - Formatted keypoints from extractKeypoints
 * @param {Number} windowSize - Number of frames in sliding window
 * @returns {Object} - Object formatted for ST-GCN++ inference
 */
export function prepareModelInput(keypointsBuffer, windowSize = 10) {
  if (!keypointsBuffer || keypointsBuffer.length < windowSize) {
    // Not enough frames yet, return empty buffer
    return null;
  }

  // Take the last 'windowSize' frames
  const recentKeypoints = keypointsBuffer.slice(-windowSize);
  
  // Format for model input, matching the expected format in the Python worker
  return {
    type: 'landmarks',
    keypoints: recentKeypoints,
    total_frames: windowSize,
    frame_dir: 'NA',
    label: 0, // Dummy label
    start_index: 0,
    modality: 'Pose',
    test_mode: true
  };
}
