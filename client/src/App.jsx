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
  const [isProcessingComplete, setIsProcessingComplete] = useState(false);
  
  // Use refs to track frame counters and implement throttling
  const frameCounter = useRef(0);
  const lastSentTime = useRef(0);
  const keypointBufferRef = useRef([]);
  const processingRef = useRef(false);
  
  // Throttling configuration
  const SEND_INTERVAL_MS = 1000; // Send data every 1 second at most
  const FRAMES_TO_BUFFER = 15; // Buffer 15 frames before sending for more responsive feedback
  const LAYOUT = 'hand_body_27'; // ST-GCN++ layout format
  const CONFIDENCE_THRESHOLD = 0.4; // Minimum confidence to consider a detection valid
  
  const handleLandmarks = useCallback((landmarksData) => {
    setLandmarks(landmarksData);
    
    // If we already have a high-confidence result or are currently processing, skip
    if (isProcessingComplete || processingRef.current) {
      return;
    }
    
    if (landmarksData && (landmarksData.hands || landmarksData.pose)) {
      // Increment frame counter
      frameCounter.current++;
      
      // Extract the keypoints in the ST-GCN++ format
      const formattedKeypoints = extractKeypoints(
        landmarksData.pose, 
        landmarksData.hands,
        LAYOUT
      );
      
      // Process this single frame immediately (no buffering on the client)
      processLandmarks({
        type: 'landmarks',
        keypoints: formattedKeypoints.keypoint  // Just send the current frame's keypoints
      });
    }
  }, [isProcessingComplete]);
  
  // Function to process landmarks and send to backend
  const processLandmarks = async (data) => {
    try {
      // Prevent multiple concurrent API calls
      if (processingRef.current) {
        return;
      }
      
      processingRef.current = true;
      setProcessingStatus('processing');
      
      // The keypoints are already properly formatted from prepareModelInput
      // We can send them directly to the API service
      console.log('Sending data to API:', data);
    
      // Use the API service to send data
      const result = await sendLandmarksToAPI(data);
      console.log('API response:', result);
      
      // Update UI with translation results
      if (result.result && result.prediction) {
        setTranslation(result.prediction);
        
        // If confidence is provided by the model
        if (result.score !== undefined) {
          setConfidence(result.score);
          
          // Store detected gesture in history if it passes threshold
          if (result.prediction !== "Unknown" && result.score > CONFIDENCE_THRESHOLD) {
            setDetectedGestures(prev => {
              const newGestures = [...prev, {
                gesture: result.prediction,
                confidence: result.score,
                timestamp: new Date().toLocaleTimeString()
              }];
              // Keep only the most recent 10 gestures
              return newGestures.slice(-10);
            });
            
            // If we have a high-confidence prediction, stop processing for a while
            if (result.score > 0.7) {
              setIsProcessingComplete(true);
              
              // Resume processing after 3 seconds to allow for new gestures
              setTimeout(() => {
                setIsProcessingComplete(false);
              }, 3000);
            }
          }
        }
      }
      
      setProcessingStatus('success');
    } catch (error) {
      console.error('Error processing landmarks:', error);
      setError(error.message);
      setProcessingStatus('error');
    } finally {
      processingRef.current = false;
    }
  };

  // Handle file upload for video processing
  const handleVideoUpload = async (file) => {
    try {
      setProcessingStatus('processing');
      processingRef.current = true;
      
      // Use the API service to upload video
      const result = await uploadVideoForProcessing(file);
      console.log('Video processing result:', result);
      
      // Update translation result
      if (result.result && result.result.result) {
        setTranslation(result.result.result);
        if (result.result.confidence) {
          setConfidence(result.result.confidence);
          
          // Add to gesture history
          if (result.result.result !== "Unknown" && result.result.confidence > CONFIDENCE_THRESHOLD) {
            setDetectedGestures(prev => {
              const newGestures = [...prev, {
                gesture: result.result.result,
                confidence: result.result.confidence,
                timestamp: new Date().toLocaleTimeString()
              }];
              return newGestures.slice(-10);
            });
          }
        }
      }
      
      setProcessingStatus('success');
    } catch (error) {
      console.error('Error uploading video:', error);
      setError(error.message);
      setProcessingStatus('error');
    } finally {
      processingRef.current = false;
    }
  };

  // Reset processing state
  const resetProcessingState = useCallback(() => {
    setIsProcessingComplete(false);
    setProcessingStatus('idle');
    setError(null);
  }, []);

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
              processingComplete={isProcessingComplete}
            />
          </div>
            {/* Right Side - Translation Results (30% width) */}
          <div className="md:w-[30%]">
            <TranslationResults
              translation={translation}
              confidence={confidence}
              processingStatus={processingStatus}
              error={error}
              detectedGestures={detectedGestures} 
            />
            
            {isProcessingComplete && (
              <div className="mt-4">
                <button 
                  onClick={resetProcessingState}
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full"
                >
                  Reset Detection
                </button>
              </div>
            )}
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