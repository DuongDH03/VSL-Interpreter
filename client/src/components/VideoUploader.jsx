import { useState, useCallback, useEffect } from 'react';
import { useMediaPipe } from '../hooks/useMediaPipe';

/**
 * Improved VideoUploader component with proper MediaPipe model lifecycle management
 * Leverages custom hook for MediaPipe handling
 */
export default function VideoUploader({ onLandmarks, onVideoUpload, processingComplete }) {
  // MediaPipe hook handles model initialization, mounting, cleanup
  const {
    videoRef, 
    canvasRef,
    isLoading,
    error,
    isCameraOn,
    handJsonData,
    poseJsonData,
    startWebcam,
    stopWebcam,
    processVideoFile
  } = useMediaPipe((landmarksData) => {
    // Only pass the landmarks to parent if not in processing complete state
    if (!processingComplete && onLandmarks) {
      onLandmarks(landmarksData);
    }
  });

  // Handle video file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Send file to parent component for API processing if provided
    if (onVideoUpload && typeof onVideoUpload === 'function') {
      onVideoUpload(file);
    }
    
    // Process the video file locally with MediaPipe
    processVideoFile(file);
  };

  return (
    <div className="flex flex-col bg-white p-6 rounded-xl shadow-md w-full">
      <h2 className="text-2xl font-bold mb-5 text-gray-800">Video Input</h2>

      {isLoading && (
        <div className="text-blue-600 text-lg mb-4 p-3 bg-blue-50 rounded-lg">Loading tracking models...</div>
      )}
      
      {error && (
        <div className="text-red-600 text-lg mb-4 p-3 bg-red-100 border border-red-400 rounded-lg">
          Error: {error}
        </div>
      )}
      
      {processingComplete && (
        <div className="text-green-600 text-lg mb-4 p-3 bg-green-50 rounded-lg border border-green-200">
          Sign detected! Click "Reset Detection" to continue recognition.
        </div>
      )}

      {/* Video and Canvas Section */}
      <div className="flex flex-col items-center w-full">
        <div className="relative w-full aspect-video rounded-lg overflow-hidden border-2 border-gray-300 bg-black">
          {/* The video element */}
          <video
            ref={videoRef}
            className="absolute top-0 left-0 w-full h-full object-contain" 
            autoPlay
            muted
            playsInline
          />
          {/* The canvas displays the video feed with landmarks drawn on it */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full object-contain"
          />
          
          {processingComplete && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-sm">
              <div className="bg-white p-4 rounded-lg text-center shadow-lg">
                <p className="font-bold text-xl text-green-700">Sign Detected!</p>
                <p className="text-gray-600">Click "Reset Detection" to continue</p>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-wrap justify-center gap-4 mt-6 w-full">
          {!isCameraOn ? (
            <button
              onClick={startWebcam}
              disabled={isLoading}
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75 transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Loading Models...' : 'Enable Webcam'}
            </button>
          ) : (
            <button
              onClick={stopWebcam}
              className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-75 transition duration-300 ease-in-out"
            >
              Turn Off Camera
            </button>
          )}

          <label className={`px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition duration-300 ease-in-out ${processingComplete || isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
            Upload Video
            <input
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              disabled={isLoading || processingComplete}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {/* Debug Information (Collapsible) */}
      <details className="mt-6 border border-gray-200 rounded-lg p-2">
        <summary className="font-semibold text-gray-700 cursor-pointer p-2">
          Debug Information (Show/Hide Landmark Data)
        </summary>
        <div className="flex flex-col md:flex-row gap-4 p-2">
          <div className="flex-1 p-3 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-2 text-sm text-gray-700">Hand Landmarks</h3>
            <pre className="bg-gray-800 text-green-400 p-3 rounded-lg overflow-auto h-[20vh] text-xs">
              <code>{handJsonData}</code>
            </pre>
          </div>
          <div className="flex-1 p-3 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-2 text-sm text-gray-700">Pose Landmarks</h3>
            <pre className="bg-gray-800 text-green-400 p-3 rounded-lg overflow-auto h-[20vh] text-xs">
              <code>{poseJsonData}</code>
            </pre>
          </div>
        </div>
      </details>
    </div>
  );
}
