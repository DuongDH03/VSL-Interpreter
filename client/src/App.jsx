import { useState, useCallback, useEffect, useRef } from 'react';
import VideoUploader from './components/VideoUploader';
import TranslationResults from './components/TranslationResults';
import LearnComponent from './components/LearnComponent';
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
  
  // Mode switching state - 'translate' or 'learn'
  const [mode, setMode] = useState('translate');
  
  // Learn mode state
  const [currentSign, setCurrentSign] = useState('16.MatTroi');
  const [isCorrect, setIsCorrect] = useState(null);

  // Use refs to track frame counters and implement throttling
  const frameCounter = useRef(0);
  const lastSentTime = useRef(0);
  const keypointBufferRef = useRef([]); // This ref seems unused with current logic
  const processingRef = useRef(false);

  // Throttling configuration
  const SEND_INTERVAL_MS = 1000; // Send data every 1 second at most
  const FRAMES_TO_BUFFER = 15; // Buffer 15 frames before sending for more responsive feedback
  const LAYOUT = 'hand_body_27'; // ST-GCN++ layout format
  const CONFIDENCE_THRESHOLD = 0.4; // Minimum confidence to consider a detection valid  // Use refs to track the latest state values to avoid closure issues
  const modeRef = useRef(mode);
  const currentSignRef = useRef(currentSign);
  const isProcessingCompleteRef = useRef(isProcessingComplete);
  
  // Keep refs updated with latest values
  useEffect(() => {
    modeRef.current = mode;
    console.log(`App: mode ref updated to ${mode}`);
  }, [mode]);
  
  useEffect(() => {
    currentSignRef.current = currentSign;
  }, [currentSign]);
  
  useEffect(() => {
    isProcessingCompleteRef.current = isProcessingComplete;
  }, [isProcessingComplete]);

  const handleLandmarks = useCallback((landmarksData) => {
    setLandmarks(landmarksData);

    // Always use the ref values for the latest state
    const currentMode = modeRef.current;
    const currentExpectedSign = currentSignRef.current;
    const isComplete = isProcessingCompleteRef.current;

    // If we already have a high-confidence result or are currently processing, skip
    if (isComplete || processingRef.current) {
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
        // Log current mode for debugging
        console.log(`Current mode in handleLandmarks: ${currentMode}`);
        
        // Add the current mode to the data payload
        const payloadData = {
          type: 'landmarks',
          keypoints: formattedKeypoints.keypoint, // Just send the current frame's keypoints
          mode: currentMode, // Send the current mode to help the server
          expectedSign: currentMode === 'learn' ? currentExpectedSign : undefined // Send the expected sign in learn mode
        };
        
        console.log('Prepared payload with mode:', payloadData.mode);
        processLandmarks(payloadData);
        lastSentTime.current = now; // Update last sent time
      }
    }
  }, []); // No dependencies as we use refs  // Function to process landmarks and send to backend
  const processLandmarks = async (data) => {
    try {
      // Prevent multiple concurrent API calls
      if (processingRef.current) {
        return;
      }

      processingRef.current = true;
      setProcessingStatus('processing'); // Set processing status at the start of API call

      // Always use the current mode from the ref
      const currentMode = modeRef.current;
      
      console.log('Current app mode (from ref):', currentMode);
      console.log('Sending data to API:', data);

      // Use the API service to send data
      const result = await sendLandmarksToAPI(data);
      console.log('API response:', result);// Update UI with translation results
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
            // Handle the prediction differently based on current mode
          const currentMode = modeRef.current;
          console.log(`Processing result in mode (from ref): ${currentMode}`);
          
          if (currentMode === 'translate') {
            // Translation mode: store in history
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
              if (score > CONFIDENCE_THRESHOLD) {
                setIsProcessingComplete(true);
  
                // Resume processing after 3 seconds to allow for new gestures
                setTimeout(() => {
                  setIsProcessingComplete(false);
                }, 3000);
              }
            }          } else if (currentMode === 'learn') {
            console.log("===== LEARN MODE PROCESSING =====");
            console.log("Current mode:", currentMode);
            console.log("Prediction:", prediction);
            console.log("Current sign:", currentSignRef.current);
            console.log("Score:", score);
            console.log("Confidence threshold:", CONFIDENCE_THRESHOLD);
            
            // Learn mode: check if prediction matches current sign
            if (score > CONFIDENCE_THRESHOLD) {              // Try both with and without prefix numbers (16.MatTroi vs MatTroi)
              const cleanPrediction = prediction.replace(/^\d+\./, '').trim();
              const cleanCurrentSign = currentSignRef.current.replace(/^\d+\./, '').trim();
                // Do more forgiving comparison - both exact match and cleaned match
              const isExactMatch = prediction.toLowerCase() === currentSignRef.current.toLowerCase();
              const isCleanMatch = cleanPrediction.toLowerCase() === cleanCurrentSign.toLowerCase();
              
              // Also check if prediction contains the sign name or vice versa
              const predictionContainsSign = prediction.toLowerCase().includes(cleanCurrentSign.toLowerCase());
              const signContainsPrediction = currentSign.toLowerCase().includes(cleanPrediction.toLowerCase());
              
              const isSignCorrect = isExactMatch || isCleanMatch || predictionContainsSign || signContainsPrediction;
                console.log("Clean prediction:", cleanPrediction);
              console.log("Clean current sign:", cleanCurrentSign);
              console.log("Is exact match:", isExactMatch);
              console.log("Is clean match:", isCleanMatch);
              console.log("Prediction contains sign:", predictionContainsSign);
              console.log("Sign contains prediction:", signContainsPrediction);
              console.log("Final result - is sign correct:", isSignCorrect);
              
              setIsCorrect(isSignCorrect);
              
              // If correct, show success for a moment then reset
              if (isSignCorrect) {
                setIsProcessingComplete(true);
                console.log("✓ CORRECT SIGN DETECTED! Setting processing complete.");
                
                // Reset after 3 seconds to allow for next attempt
                setTimeout(() => {
                  setIsProcessingComplete(false);
                  setIsCorrect(null);
                }, 3000);
              } else {
                console.log("✗ INCORRECT SIGN. Try again.");
              }
            } else {
              console.log("Score too low to determine match:", score, "< threshold", CONFIDENCE_THRESHOLD);
            }
          }
        }
      } else {
        console.log('No prediction found in API response:', result);
      }
      setProcessingStatus('success'); // Set success status only after translation and confidence are updated
      console.log('Current state after API call:', { 
        translation: prediction, 
        confidence: score, 
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
    
    // Reset learn mode state if in learn mode
    if (mode === 'learn') {
      setIsCorrect(null);
    }  }, [mode]);

  // Log current state values whenever they change for debugging
  useEffect(() => {
    console.log('App state updated:', {
      mode,
      translation,
      confidence,
      processingStatus,
      error,
      isProcessingComplete,
      currentSign: mode === 'learn' ? currentSign : null,
      isCorrect: mode === 'learn' ? isCorrect : null,
      detectedGestures: detectedGestures.length
    });
  }, [mode, translation, confidence, processingStatus, error, isProcessingComplete, currentSign, isCorrect, detectedGestures]);
  return (
    <div className="app-container w-full h-full min-h-screen bg-gray-100">      {/* Debug Info Banner */}
      <div className="bg-yellow-100 text-black p-2 text-center text-xs">
        <p>
          <strong>DEBUG INFO:</strong> 
          Mode: <span className="font-mono bg-white p-1 rounded">{mode}</span> | 
          Current Sign: <span className="font-mono bg-white p-1 rounded">{currentSign}</span> | 
          Processing: <span className="font-mono bg-white p-1 rounded">{processingStatus}</span> | 
          Complete: <span className="font-mono bg-white p-1 rounded">{String(isProcessingComplete)}</span>
        </p>
      </div>
      
      <header className="bg-blue-600 text-white p-4 shadow-lg">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold text-center">Sign Language Interpreter</h1>
          <p className="text-center mt-2">Using MediaPipe and ST-GCN++ for real-time sign language translation</p>
          
          {/* Mode Toggle Buttons */}
          <div className="flex justify-center mt-4 gap-4">            <button 
              onClick={() => {
                console.log("Switching to translate mode");
                setMode('translate');
                // Reset any learn-specific state
                setIsCorrect(null);
              }} 
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                mode === 'translate' 
                  ? 'bg-white text-blue-600' 
                  : 'bg-blue-700 text-white hover:bg-blue-800'
              }`}
            >
              Translate
            </button>
            <button 
              onClick={() => {
                console.log("Switching to learn mode");
                setMode('learn');
                // Reset any translate-specific state
                setTranslation('');
                setConfidence(0);
              }} 
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                mode === 'learn' 
                  ? 'bg-white text-blue-600' 
                  : 'bg-blue-700 text-white hover:bg-blue-800'
              }`}
            >
              Learn
            </button>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <div className="flex flex-col md:flex-row gap-6">          {/* Left Side - Video Uploader (70% width) */}
          <div className="md:w-[70%]">
            <VideoUploader
              onLandmarks={handleLandmarks}
              onVideoUpload={handleVideoUpload}
              processingComplete={isProcessingComplete}
              mode={mode}
            />
          </div>
          {/* Right Side - Results (30% width) */}
          <div className="md:w-[30%]">
            {mode === 'translate' ? (
              <TranslationResults
                translation={translation}
                confidence={confidence}
                processingStatus={processingStatus}
                error={error}
                detectedGestures={detectedGestures}
              />
            ) : (
              <LearnComponent
                currentSign={currentSign}
                isCorrect={isCorrect}
                processingStatus={processingStatus}
                error={error}
              />
            )}

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