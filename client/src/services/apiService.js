/**
 * API service to handle communications with the server
 */

const API_URL = 'http://localhost:3001/api';

/**
 * Send landmarks data to the server for live video translation
 * 
 * @param {Object} landmarksData - Object containing hand and pose landmarks
 * @param {Array} [keypointBuffer=[]] - Buffer of previously processed keypoints for inference
 * @param {Boolean} [runInference=false] - Whether to run inference on the server
 * @returns {Promise} - Promise that resolves with the translation results
 */
export async function sendLandmarksToAPI(landmarksData, keypointBuffer = [], runInference = false) {
  try {
    console.log('API request payload:', JSON.stringify(landmarksData, null, 2));
    const response = await fetch(`${API_URL}/translate/live-video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(landmarksData),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending landmarks to API:', error);
    throw error;
  }
}

/**
 * Upload a video file for processing
 * 
 * @param {File} file - The video file to upload
 * @returns {Promise} - Promise that resolves with the translation results
 */
export async function uploadVideoForProcessing(file) {
  try {
    const formData = new FormData();
    formData.append('video', file);
    
    const response = await fetch(`${API_URL}/translate/video`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Failed to process video');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading video:', error);
    throw error;
  }
}

/**
 * Learn a new sign/gesture 
 * 
 * @param {Object} data - Data containing landmarks and label for learning
 * @returns {Promise} - Promise that resolves with the learning results
 */
export async function learnNewSign(data) {
  try {
    const response = await fetch(`${API_URL}/learn/new-sign`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${errorText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error learning new sign:', error);
    throw error;
  }
}
