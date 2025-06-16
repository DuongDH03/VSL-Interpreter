import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Custom hook for handling MediaPipe models lifecycle
 * Centralizes the complexity of managing MediaPipe models, rendering, and cleanup
 */
export function useMediaPipe(onLandmarks) {
  // Set up necessary refs and state
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [handJsonData, setHandJsonData] = useState('No hands detected yet.');
  const [poseJsonData, setPoseJsonData] = useState('No pose detected yet.');

  // Store callback in a ref to ensure we always use the latest version
  const onLandmarksRef = useRef(onLandmarks);
  
  // Keep the ref updated with the latest callback
  useEffect(() => {
    onLandmarksRef.current = onLandmarks;
  }, [onLandmarks]);

  // Refs for MediaPipe models that should persist between renders
  const handsRef = useRef(null);
  const poseRef = useRef(null);
  const cameraRef = useRef(null);
  
  // Critical for correct lifecycle management
  const mountedRef = useRef(true);
  
  // Store the latest results for hands and pose
  const lastHandsResults = useRef(null);
  const lastPoseResults = useRef(null);
  
  // Function to render both hand and pose results on canvas
  const renderResults = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !mountedRef.current) return;
    
    const canvasCtx = canvasRef.current.getContext('2d');
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Apply mirroring transformation once for the entire frame
    canvasCtx.translate(canvasRef.current.width, 0);
    canvasCtx.scale(-1, 1);

    // Draw the video frame onto the canvas
    canvasCtx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    
    const handsResults = lastHandsResults.current;
    const poseResults = lastPoseResults.current;

    // --- Draw Hand Landmarks ---
    if (handsResults && handsResults.multiHandLandmarks && handsResults.multiHandLandmarks.length > 0) {
      setHandJsonData(JSON.stringify(handsResults.multiHandLandmarks, null, 2));

      for (const landmarks of handsResults.multiHandLandmarks) {
        // Draw dots for each hand landmark
        for (const landmark of landmarks) {
          canvasCtx.beginPath();
          canvasCtx.arc(landmark.x * canvasRef.current.width, landmark.y * canvasRef.current.height, 5, 0, 2 * Math.PI);
          canvasCtx.fillStyle = '#FF0000'; // Red color for hand landmarks
          canvasCtx.fill();
        }

        // Draw connections between hand landmarks
        const handConnections = [
          [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
          [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
          [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
          [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
          [0, 17], [17, 18], [18, 19], [19, 20], // Pinky finger
          [0, 5], [5, 9], [9, 13], [13, 17], [17, 0] // Palm base
        ];
        canvasCtx.strokeStyle = '#00FF00'; // Green color for hand connections
        canvasCtx.lineWidth = 2;
        for (const connection of handConnections) {
          const start = landmarks[connection[0]];
          const end = landmarks[connection[1]];
          canvasCtx.beginPath();
          canvasCtx.moveTo(start.x * canvasRef.current.width, start.y * canvasRef.current.height);
          canvasCtx.lineTo(end.x * canvasRef.current.width, end.y * canvasRef.current.height);
          canvasCtx.stroke();
        }
      }
    } else {
      setHandJsonData('No hands detected.');
    }

    // --- Draw Pose Landmarks ---
    if (poseResults && poseResults.poseLandmarks) {
      setPoseJsonData(JSON.stringify(poseResults.poseLandmarks, null, 2));

      // Draw pose landmarks
      for (const landmark of poseResults.poseLandmarks) {
        const x = landmark.x * canvasRef.current.width;
        const y = landmark.y * canvasRef.current.height;
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
        canvasCtx.fillStyle = '#4285F4'; // Blue color for pose landmarks
        canvasCtx.fill();
      }

      // Draw pose connections using drawing_utils (if available) or fallback
      if (window.drawConnectors && window.POSE_CONNECTIONS) {
        window.drawConnectors(
          canvasCtx,
          poseResults.poseLandmarks,
          window.POSE_CONNECTIONS,
          { color: '#FFA500', lineWidth: 2 } // Orange color for pose connections
        );
      } else {
        // Fallback for pose connections (basic set)
        const poseConnections = [
          [11, 12], // Shoulders
          [11, 13], [13, 15], // Left arm
          [12, 14], [14, 16], // Right arm
          [11, 23], [12, 24], // Torso to hips
          [23, 25], [25, 27], // Left leg
          [24, 26], [26, 28] // Right leg
        ];
        canvasCtx.strokeStyle = '#FFA500'; // Orange color for fallback connections
        canvasCtx.lineWidth = 2;
        for (const connection of poseConnections) {
          const start = poseResults.poseLandmarks[connection[0]];
          const end = poseResults.poseLandmarks[connection[1]];
          if (start && end) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(start.x * canvasRef.current.width, start.y * canvasRef.current.height);
            canvasCtx.lineTo(end.x * canvasRef.current.width, end.y * canvasRef.current.height);
            canvasCtx.stroke();
          }
        }
      }
    } else {
      setPoseJsonData('No pose detected.');
    }
      // Call onLandmarks with combined data if available
    if (mountedRef.current) {      // Always use the latest callback function from the ref
      if (onLandmarksRef.current) {
        // Create landmarks data object
        const landmarksData = {
          hands: handsResults?.multiHandLandmarks || null,
          pose: poseResults?.poseLandmarks || null
        };
        
        // Call the latest callback
        onLandmarksRef.current(landmarksData);
      }}

    canvasCtx.restore();
  }, []); // Remove onLandmarks from deps since we use ref

  // Combined callback for MediaPipe results from both models
  const onFrameResults = useCallback((handsResults, poseResults) => {
    // Skip processing if component is unmounting or refs invalid
    if (!mountedRef.current || !videoRef.current || !canvasRef.current) return;
    
    if (handsResults) {
      lastHandsResults.current = handsResults;
      console.log("Received hand results:", 
        handsResults.multiHandLandmarks?.length > 0 ? 
        `${handsResults.multiHandLandmarks.length} hands detected` : 
        "No hands detected");
    }
    
    if (poseResults) {
      lastPoseResults.current = poseResults;
      console.log("Received pose results:",
        poseResults.poseLandmarks ? 
        "Pose detected" : 
        "No pose detected");
    }
    
    // Render with the latest available data for both models
    renderResults();
  }, [renderResults]);

  // Initialize MediaPipe models once on mount
  useEffect(() => {
    console.log("MediaPipe hook initializing");
    
    // Set as mounted
    mountedRef.current = true;
    
    // Helper function to load scripts if needed
    const loadScript = (src, id) => {
      return new Promise((resolve, reject) => {
        if (document.getElementById(id)) {
          resolve();
          return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.id = id;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
        document.head.appendChild(script);
      });
    };
    
    // Function to initialize MediaPipe libraries
    async function initializeMediaPipe() {
      try {
        setIsLoading(true);
        
        // Scripts should be loaded in index.html, but check anyway
        if (!window.Hands || !window.Pose || !window.Camera) {
          console.log("Loading MediaPipe scripts dynamically");
          await Promise.all([
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js', 'mediapipe-hands'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js', 'mediapipe-pose'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3/camera_utils.js', 'mediapipe-camera'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3/drawing_utils.js', 'mediapipe-drawing')
          ]);
        }
        
        if (!window.Hands || !window.Pose || !window.Camera) {
          throw new Error("MediaPipe libraries could not be loaded");
        }
        
        console.log("Creating MediaPipe model instances");
        
        // Initialize Hands
        const hands = new window.Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
          }
        });
        
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        // Set up the results callback with a guard for component mounting
        hands.onResults((results) => {
          if (mountedRef.current) {
            onFrameResults(results, null);
          }
        });
        
        // Store in ref
        handsRef.current = hands;
        
        // Initialize Pose
        const pose = new window.Pose({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`;
          }
        });
        
        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        // Set up the results callback with a guard for component mounting
        pose.onResults((results) => {
          if (mountedRef.current) {
            onFrameResults(null, results);
          }
        });
        
        // Store in ref
        poseRef.current = pose;
        
        if (mountedRef.current) {
          setIsLoading(false);
          console.log("MediaPipe models initialized successfully");
        }
      } catch (err) {
        console.error("Failed to initialize MediaPipe:", err);
        if (mountedRef.current) {
          setError(`Failed to initialize tracking: ${err.message}`);
          setIsLoading(false);
        }
      }
    }
    
    // Start initialization
    initializeMediaPipe();
    
    // Cleanup on unmount
    return () => {
      console.log("MediaPipe hook cleaning up");
      mountedRef.current = false;
      
      // Stop camera if active
      if (cameraRef.current) {
        try {
          console.log("Stopping camera");
          cameraRef.current.stop();
          cameraRef.current = null;
        } catch (err) {
          console.error("Error stopping camera:", err);
        }
      }
      
      // Stop media streams
      if (stream) {
        try {
          stream.getTracks().forEach(track => track.stop());
        } catch (err) {
          console.error("Error stopping media stream:", err);
        }
      }
      
      // Clean up model instances
      try {
        if (handsRef.current) {
          console.log("Closing hands model");
          handsRef.current.close();
          handsRef.current = null;
        }
        
        if (poseRef.current) {
          console.log("Closing pose model");
          poseRef.current.close();
          poseRef.current = null;
        }
        
        // Clear result refs
        lastHandsResults.current = null;
        lastPoseResults.current = null;
      } catch (err) {
        console.error("Error during model cleanup:", err);
      }
    };
  }, []); // Empty dependency array ensures this runs only once

  // Function to set up the camera utility when webcam is requested
  const setupCamera = useCallback(() => {
    if (!videoRef.current || !poseRef.current || !handsRef.current) {
      console.error("Cannot set up camera - video or models not ready");
      return false;
    }
    
    if (!cameraRef.current && window.Camera) {
      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          if (!mountedRef.current || !videoRef.current) return;
          
          try {
            if (poseRef.current) {
              await poseRef.current.send({ image: videoRef.current });
            }
            
            if (handsRef.current) {
              await handsRef.current.send({ image: videoRef.current });
            }
          } catch (err) {
            console.error("Error in camera frame processing:", err);
          }
        },
        width: 640,
        height: 480
      });
      
      cameraRef.current = camera;
      console.log("Camera utility initialized");
      return true;
    }
    
    return !!cameraRef.current;
  }, []);

  // Function to start webcam
  const startWebcam = useCallback(async () => {
    console.log("Starting webcam...");
    
    // Make sure we have everything we need
    const cameraReady = setupCamera();
    
    if (!cameraReady) {
      setError("Failed to initialize camera utility");
      return;
    }
    
    try {
      // Request media stream
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        
        // Set up metadata loaded handler to ensure dimensions are right
        videoRef.current.onloadedmetadata = () => {
          if (!mountedRef.current) return;
          
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
          
          videoRef.current.play().then(() => {
            if (!mountedRef.current) return;
            
            if (cameraRef.current) {
              cameraRef.current.start();
              console.log("Camera processing started");
              setIsCameraOn(true);
            }
          }).catch(err => {
            console.error("Error playing video:", err);
            setError(`Could not play video: ${err.message}`);
          });
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError(`Could not access webcam: ${err.message}`);
    }
  }, [setupCamera]);

  // Function to stop webcam
  const stopWebcam = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    
    if (cameraRef.current) {
      cameraRef.current.stop();
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.pause();
    }
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    
    setHandJsonData('Webcam stopped. No hands detected.');
    setPoseJsonData('Webcam stopped. No pose detected.');
    setIsCameraOn(false);
  }, [stream]);

  // Function to process video files
  const processVideoFile = useCallback((file) => {
    if (!file) return;
    
    // Create blob URL for the file
    const videoUrl = URL.createObjectURL(file);
    
    // Stop any active webcam
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    
    if (cameraRef.current) {
      cameraRef.current.stop();
    }
    
    // Set up video element with the file
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = videoUrl;
      videoRef.current.load();
      
      videoRef.current.onloadedmetadata = () => {
        if (!mountedRef.current) return;
        
        if (canvasRef.current && videoRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
        
        videoRef.current.play().catch(err => {
          console.error("Error playing video file:", err);
          setError(`Could not play video file: ${err.message}`);
        });
        
        setIsCameraOn(true);
        
        // Process video frames
        const processVideoFrame = async () => {
          if (!mountedRef.current) return;
          
          if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
            try {
              if (handsRef.current) {
                await handsRef.current.send({ image: videoRef.current });
              }
              
              if (poseRef.current) {
                await poseRef.current.send({ image: videoRef.current });
              }
              
              requestAnimationFrame(processVideoFrame);
            } catch (err) {
              console.error("Error processing video frame:", err);
            }
          } else if (videoRef.current && videoRef.current.ended) {
            setHandJsonData('Video ended. No hands detected.');
            setPoseJsonData('Video ended. No pose detected.');
            setIsCameraOn(false);
          }
        };
        
        requestAnimationFrame(processVideoFrame);
      };
    }
  }, [stream]);

  return {
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
  };
}
