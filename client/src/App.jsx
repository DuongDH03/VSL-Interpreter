import { useState, useCallback, useEffect, useRef } from 'react';
import VideoUploader from './components/VideoUploader';
import TranslationResults from './components/TranslationResults';
import { sendLandmarksToAPI, uploadVideoForProcessing } from './services/apiService';
import { extractKeypoints, prepareModelInput } from './utils/keypointProcessor';
import './App.css';

function App() {
  const [landmarks, setLandmarks] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [translation, setTranslation] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);
  const [detectedGestures, setDetectedGestures] = useState([]);
  
  // Use refs to track frame counters and implement throttling
  const frameCounter = useRef(0);
  const lastSentTime = useRef(0);
  const keypointBufferRef = useRef([]);
    // Throttling configuration
  const SEND_INTERVAL_MS = 1000; // Send data every 1 second at most
  const FRAMES_TO_BUFFER = 15; // Buffer 15 frames before sending for more responsive feedback
  const LAYOUT = 'hand_body_27'; // ST-GCN++ layout format
  
  const handleLandmarks = useCallback((landmarksData) => {
    setLandmarks(landmarksData);
    
    if (landmarksData && (landmarksData.hands || landmarksData.pose)) {
      // Increment frame counter
      frameCounter.current++;
      
      // Extract the keypoints in the ST-GCN++ format
      const formattedKeypoints = extractKeypoints(
        landmarksData.pose, 
        landmarksData.hands,
        LAYOUT
      );
      
      // Add keypoints to buffer
      keypointBufferRef.current.push(formattedKeypoints.keypoint[0]);
      
      // Keep buffer at desired size
      if (keypointBufferRef.current.length > FRAMES_TO_BUFFER * 2) {
        keypointBufferRef.current.shift();
      }
      
      // Check if we should send data based on time and buffer size
      const now = Date.now();
      const timeSinceLastSend = now - lastSentTime.current;
      
      if (keypointBufferRef.current.length >= FRAMES_TO_BUFFER && 
          timeSinceLastSend >= SEND_INTERVAL_MS) {
        // Reset counter and send landmarks
        frameCounter.current = 0;
        lastSentTime.current = now;
        
        // Prepare model input with buffered keypoints
        const modelInput = prepareModelInput(keypointBufferRef.current, FRAMES_TO_BUFFER);
        
        // Only send if we have enough data
        if (modelInput) {
          processLandmarks({
            keypoints: modelInput,
            raw_data: {
              hand_landmarks: landmarksData.hands,
              pose_landmarks: landmarksData.pose
            }
          });
        }
      }
    }
  }, []);
    // Function to process landmarks and send to backend
  const processLandmarks = async (data) => {
    try {
      setProcessingStatus('processing');
      

      // Make sure the keypoints are in the right format for the Python worker
      const requestData = {
        type: 'landmarks',  // Use the command type expected by the worker
        keypoints: data.keypoints.keypoint[0]  // Send the correct format (V, C) array
      };

      console.log('Formatted request data:', requestData);
    
      // Use the API service to send data
      const result = await sendLandmarksToAPI(data);
      console.log('API response:', result);
      
      // Update UI with translation results
      if (result.result && result.result.result) {
        setTranslation(result.result.result);
        // If confidence is provided by the model
        if (result.result.confidence) {
          setConfidence(result.result.confidence);
        }
      }
      
      setProcessingStatus('success');
    } catch (error) {
      console.error('Error processing landmarks:', error);
      setError(error.message);
      setProcessingStatus('error');
    }
  };

  // Handle file upload for video processing
  const handleVideoUpload = async (file) => {
    try {
      setProcessingStatus('processing');
      
      // Use the API service to upload video
      const result = await uploadVideoForProcessing(file);
      console.log('Video processing result:', result);
      
      // Update translation result
      if (result.result && result.result.result) {
        setTranslation(result.result.result);
        if (result.result.confidence) {
          setConfidence(result.result.confidence);
        }
      }
      
      setProcessingStatus('success');
    } catch (error) {
      console.error('Error uploading video:', error);
      setError(error.message);
      setProcessingStatus('error');
    }
  };

  return (
    <div className="app-container w-full h-full min-h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-4 shadow-lg">
        <h1 className="text-3xl font-bold text-center">Sign Language Interpreter</h1>
        <p className="text-center mt-2">Using MediaPipe and ST-GCN++ for real-time sign language translation</p>
      </header>
      
      <main className="container mx-auto p-4">
        <div className="flex flex-col md:flex-row gap-6">
          {/* Left Side - Video Uploader (70% width) */}
          <div className="md:w-[70%]">
            <VideoUploader 
              onLandmarks={handleLandmarks} 
              onVideoUpload={handleVideoUpload} 
            />
          </div>
            {/* Right Side - Translation Results (30% width) */}
          <div className="md:w-[30%]">
            <TranslationResults
              translation={translation}
              confidence={confidence}
              processingStatus={processingStatus}
              error={error}
            />
          </div>
        </div>
      </main>
      
      <footer className="bg-gray-700 text-white text-center p-4 mt-8">
        <p>© 2025 Sign Language Interpreter - Powered by MediaPipe and ST-GCN++</p>
      </footer>
    </div>
  );
}
export default App;