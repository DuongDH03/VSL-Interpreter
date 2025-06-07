import { useState, useCallback, useEffect, useRef } from 'react';
import VideoUploader from './components/VideoUploader';
import './App.css';

function App() {
  const [landmarks, setLandmarks] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [translation, setTranslation] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);
  
  // Use refs to track frame counters and implement throttling
  const frameCounter = useRef(0);
  const lastSentTime = useRef(0);
  const bufferRef = useRef([]);
  
  // Throttling configuration
  const SEND_INTERVAL_MS = 2000; // Send data every 2 seconds at most
  const FRAMES_TO_BUFFER = 30; // Buffer 30 frames before sending
  
  const handleLandmarks = useCallback((landmarksData) => {
    setLandmarks(landmarksData);
    
    if (landmarksData && (landmarksData.hands || landmarksData.pose)) {
      // Increment frame counter
      frameCounter.current++;
      
      // Add landmarks to buffer
      bufferRef.current.push(landmarksData);
      
      // Keep buffer at desired size
      if (bufferRef.current.length > FRAMES_TO_BUFFER) {
        bufferRef.current.shift();
      }
      
      // Check if we should send data based on time and buffer size
      const now = Date.now();
      const timeSinceLastSend = now - lastSentTime.current;
      
      if (bufferRef.current.length >= FRAMES_TO_BUFFER && 
          timeSinceLastSend >= SEND_INTERVAL_MS) {
        // Reset counter and send landmarks
        frameCounter.current = 0;
        lastSentTime.current = now;
        
        // Send the buffered landmarks
        sendLandmarksToAPI({
          hand_landmarks: landmarksData.hands,
          pose_landmarks: landmarksData.pose
        });
      }
    }
  }, []);
  
  // Function to send landmarks to backend
  const sendLandmarksToAPI = async (landmarksData) => {
    try {
      setProcessingStatus('processing');
      
      // Send to server - be sure to use the correct server URL in production
      const response = await fetch('http://localhost:3001/api/translate/live-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(landmarksData),
      });
      
      if (response.ok) {
        const result = await response.json();
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
      } else {
        const errorText = await response.text();
        throw new Error(`API error: ${errorText}`);
      }
    } catch (error) {
      console.error('Error sending landmarks to API:', error);
      setError(error.message);
      setProcessingStatus('error');
    }
  };

  // Handle file upload for video processing
  const handleVideoUpload = async (file) => {
    // Create form data
    const formData = new FormData();
    formData.append('video', file);
    
    try {
      setProcessingStatus('processing');
      
      const response = await fetch('http://localhost:3001/api/translate/video', {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Video processing result:', result);
        
        // Update translation result
        if (result.result && result.result.result) {
          setTranslation(result.result.result);
        }
        
        setProcessingStatus('success');
      } else {
        throw new Error('Failed to process video');
      }
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
        <VideoUploader onLandmarks={handleLandmarks} />
        
        {/* Translation Results */}
        <section className="mt-8 bg-white rounded-xl shadow-md p-6">
          <h2 className="text-2xl font-bold text-gray-700 mb-4">Translation Results</h2>
          
          {translation ? (
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <h3 className="text-lg font-semibold text-gray-800">Detected Sign:</h3>
              <p className="text-3xl font-bold text-blue-700 my-2">{translation}</p>
              
              {confidence > 0 && (
                <div className="mt-2">
                  <p className="text-sm text-gray-600">Confidence: {(confidence * 100).toFixed(1)}%</p>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          ) : processingStatus === 'processing' ? (
            <p className="text-gray-600">Analyzing sign language...</p>
          ) : (
            <p className="text-gray-600">No sign language detected yet. Try using the webcam or uploading a video.</p>
          )}
        </section>
      </main>
      
      {/* Status Notifications */}
      {processingStatus === 'processing' && (
        <div className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg">
          Processing sign language...
        </div>
      )}
      {processingStatus === 'success' && (
        <div className="fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg">
          Translation complete!
        </div>
      )}
      {processingStatus === 'error' && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg">
          Error: {error || 'Failed to process sign language'}
        </div>
      )}
      
      <footer className="bg-gray-700 text-white text-center p-4 mt-8">
        <p>© 2025 Sign Language Interpreter - Powered by MediaPipe and ST-GCN++</p>
      </footer>
    </div>
  );
}
export default App;