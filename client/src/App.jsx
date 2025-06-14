import { useState, useCallback, useEffect, useRef } from 'react';
import VideoUploader from './components/VideoUploader';
import TranslationResults from './components/TranslationResults';
import { sendLandmarksToAPI, uploadVideoForProcessing } from './services/apiService';
import { extractKeypoints } from './utils/keypointProcessor'; // Removed prepareModelInput as it's not used here directly
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
  const keypointBufferRef = useRef([]); // This ref seems unused with current logic
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
      // Increment frame counter (this is only for potential future buffering logic if needed)
      frameCounter.current++;

      // Extract the keypoints in the ST-GCN++ format
      const formattedKeypoints = extractKeypoints(
        landmarksData.pose,
        landmarksData.hands,
        LAYOUT
      );

      // Process this single frame immediately (no buffering on the client)
      // Only process if enough time has passed since the last send, to avoid overwhelming the backend
      const now = Date.now();
      if (now - lastSentTime.current > SEND_INTERVAL_MS) {
        processLandmarks({
          type: 'landmarks',
          keypoints: formattedKeypoints.keypoint // Just send the current frame's keypoints
        });
        lastSentTime.current = now; // Update last sent time
      }
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
      setProcessingStatus('processing'); // Set processing status at the start of API call

      console.log('Sending data to API:', data);

      // Use the API service to send data
      const result = await sendLandmarksToAPI(data);
      console.log('API response:', result);      // Update UI with translation results
      // First, check the structure of the response - it might be nested in result property
      let prediction = null;
      let score = null;
      
      if (result.result && result.result.prediction) {
        // Format 1: prediction nested in result
        prediction = result.result.prediction;
        score = result.result.score !== undefined ? result.result.score : null;
        console.log('Setting translation from nested result:', prediction);
      } else if (result.prediction) {
        // Format 2: prediction at top level
        prediction = result.prediction;
        score = result.score !== undefined ? result.score : null;
        console.log('Setting translation from top level:', prediction);
      }

      // Handle buffering state
      if (prediction === 'Buffering') {
        // Show buffering status but don't update the translation
        setProcessingStatus('processing');
        console.log('Buffering frames:', result.result?.message || 'No message');
        return;
      }
      
      // If we found a valid prediction, update the UI
      if (prediction) {
        setTranslation(prediction);
        
        if (score !== null) {
          setConfidence(score);
          console.log('Setting confidence to:', score);
          
          // Store detected gesture in history if it passes threshold
          if (prediction !== "Unknown" && score > CONFIDENCE_THRESHOLD) {
            setDetectedGestures(prev => {
              const newGestures = [...prev, {
                gesture: prediction,
                confidence: score,
                timestamp: new Date().toLocaleTimeString()
              }];
              // Keep only the most recent 10 gestures
              return newGestures.slice(-10);
            });

            // If we have a high-confidence prediction, stop processing for a while
            if (score > 0.7) {
              setIsProcessingComplete(true);

              // Resume processing after 3 seconds to allow for new gestures
              setTimeout(() => {
                setIsProcessingComplete(false);
              }, 3000);
            }
          }
        }
      } else {
        console.log('No prediction found in API response:', result);
      }
      setProcessingStatus('success'); // Set success status only after translation and confidence are updated
      console.log('Current state after API call:', { 
        translation, 
        confidence, 
        processingStatus: 'success',
        error
      });
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
      console.log('Video processing result:', result);      // Handle various API response formats
      console.log('Full video processing result:', result);
      
      // Extract prediction and confidence from any of the possible response formats
      let prediction = null;
      let score = null;
      
      // Format 1: Double nested result (result.result.result)
      if (result.result && result.result.result) {
        prediction = result.result.result;
        score = result.result.confidence !== undefined ? result.result.confidence : null;
        console.log('Setting translation from double-nested result:', prediction);
      } 
      // Format 2: Single nested result (result.result.prediction)
      else if (result.result && result.result.prediction) {
        prediction = result.result.prediction;
        score = result.result.score !== undefined ? result.result.score : null;
        console.log('Setting translation from nested result:', prediction);
      }
      // Format 3: Direct result fields
      else if (result.prediction) {
        prediction = result.prediction;
        score = result.score !== undefined ? result.score : null;
        console.log('Setting translation from top level:', prediction);
      }
      
      // Handle buffering state (if applicable to video upload)
      if (prediction === 'Buffering') {
        // Show buffering status but don't update the translation
        console.log('Buffering video frames');
        return;
      }
      
      // If we found a valid prediction, update the UI
      if (prediction) {
        setTranslation(prediction);
        
        if (score !== null) {
          setConfidence(score);
          console.log('Setting confidence to:', score);
          
          // Add to gesture history (if above threshold)
          if (prediction !== "Unknown" && score > CONFIDENCE_THRESHOLD) {
            setDetectedGestures(prev => {
              const newGestures = [...prev, {
                gesture: prediction,
                confidence: score,
                timestamp: new Date().toLocaleTimeString()
              }];
              return newGestures.slice(-10);
            });
          }
        }
      } else {
        console.log('No valid prediction found in any response format:', result);
      }
      setProcessingStatus('success'); // Set success status only after translation and confidence are updated
      console.log('Current state after video upload:', { 
        translation, 
        confidence, 
        processingStatus: 'success',
        error
      });
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
    setTranslation(''); // Clear translation on reset
    setConfidence(0); // Clear confidence on reset
    setError(null);
  }, []);
  // Log current state values whenever they change for debugging
  useEffect(() => {
    console.log('App state updated:', {
      translation,
      confidence,
      processingStatus,
      error,
      isProcessingComplete,
      detectedGestures: detectedGestures.length
    });
  }, [translation, confidence, processingStatus, error, isProcessingComplete, detectedGestures]);

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