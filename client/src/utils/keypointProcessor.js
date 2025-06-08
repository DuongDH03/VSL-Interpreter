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
export function extractKeypoints(poseData, handsData, layout = 'hand_body_27') {
  const numNodes = layout === 'hand_body_27' ? 27 : 29;
  
  // Initialize keypoints array with zeros (M=1 person, V=nodes, C=3 coords)
  const keypoints = Array(1).fill().map(() => 
    Array(numNodes).fill().map(() => [0, 0, 0])
  );

  // Process pose landmarks if available
  if (poseData && poseData.length) {
    const landmarks = poseData;
    
    if (layout === 'hand_body_27') {
      // 7 body keypoints in hand_body_27: NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW
      const bodyIndices = [0, 2, 5, 11, 12, 13, 14];
      
      bodyIndices.forEach((mpIdx, i) => {
        if (landmarks[mpIdx]) {
          keypoints[0][i][0] = landmarks[mpIdx].x;
          keypoints[0][i][1] = landmarks[mpIdx].y;
          keypoints[0][i][2] = landmarks[mpIdx].visibility || 1.0;
        }
      });
      
      // Left wrist - index 7
      if (landmarks[15]) {
        keypoints[0][7][0] = landmarks[15].x;
        keypoints[0][7][1] = landmarks[15].y;
        keypoints[0][7][2] = landmarks[15].visibility || 1.0;
      }
      
      // Right wrist - index 17
      if (landmarks[16]) {
        keypoints[0][17][0] = landmarks[16].x;
        keypoints[0][17][1] = landmarks[16].y;
        keypoints[0][17][2] = landmarks[16].visibility || 1.0;
      }
    } else if (layout === 'hand_body_29') {
      // 9 body keypoints in hand_body_29 (includes mouth points)
      const bodyIndices = [0, 2, 5, 9, 10, 11, 12, 13, 14];
      
      bodyIndices.forEach((mpIdx, i) => {
        if (landmarks[mpIdx]) {
          keypoints[0][i][0] = landmarks[mpIdx].x;
          keypoints[0][i][1] = landmarks[mpIdx].y;
          keypoints[0][i][2] = landmarks[mpIdx].visibility || 1.0;
        }
      });
      
      // Left wrist - index 9
      if (landmarks[15]) {
        keypoints[0][9][0] = landmarks[15].x;
        keypoints[0][9][1] = landmarks[15].y;
        keypoints[0][9][2] = landmarks[15].visibility || 1.0;
      }
      
      // Right wrist - index 19
      if (landmarks[16]) {
        keypoints[0][19][0] = landmarks[16].x;
        keypoints[0][19][1] = landmarks[16].y;
        keypoints[0][19][2] = landmarks[16].visibility || 1.0;
      }
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
        if (layout === 'hand_body_27') {
          fingerIndices.forEach((mpIdx, i) => {
            keypoints[0][8 + i][0] = handLandmarks[mpIdx].x;
            keypoints[0][8 + i][1] = handLandmarks[mpIdx].y;
            keypoints[0][8 + i][2] = 1.0; // MediaPipe hands don't have visibility
          });
        } else if (layout === 'hand_body_29') {
          fingerIndices.forEach((mpIdx, i) => {
            keypoints[0][10 + i][0] = handLandmarks[mpIdx].x;
            keypoints[0][10 + i][1] = handLandmarks[mpIdx].y;
            keypoints[0][10 + i][2] = 1.0;
          });
        }
      } else {
        // Right hand in real world (MediaPipe "Left")
        if (layout === 'hand_body_27') {
          fingerIndices.forEach((mpIdx, i) => {
            keypoints[0][18 + i][0] = handLandmarks[mpIdx].x;
            keypoints[0][18 + i][1] = handLandmarks[mpIdx].y;
            keypoints[0][18 + i][2] = 1.0;
          });
        } else if (layout === 'hand_body_29') {
          fingerIndices.forEach((mpIdx, i) => {
            keypoints[0][20 + i][0] = handLandmarks[mpIdx].x;
            keypoints[0][20 + i][1] = handLandmarks[mpIdx].y;
            keypoints[0][20 + i][2] = 1.0;
          });
        }
      }
    });
  }

  return {
    keypoint: keypoints,
    layout
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
  
  // Format for model input, similar to create_fake_anno in the Python code
  return {
    keypoint: recentKeypoints,
    total_frames: windowSize,
    frame_dir: 'NA',
    label: 0, // Dummy label
    start_index: 0,
    modality: 'Pose',
    test_mode: true
  };
}
