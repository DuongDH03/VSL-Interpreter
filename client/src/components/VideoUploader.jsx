import { useState, useRef, useEffect, useCallback } from 'react';

export default function MediaPipeVideoUploader({ onLandmarks }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [hands, setHands] = useState(null);
  const [pose, setPose] = useState(null);
  const [camera, setCamera] = useState(null);
  const [handJsonData, setHandJsonData] = useState('No hands detected yet.');
  const [poseJsonData, setPoseJsonData] = useState('No pose detected yet.');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false); // Added: State to track camera status

  // Store MediaPipe model instances in refs so they persist between renders
  const handsRef = useRef(null);
  const poseRef = useRef(null);
  const cameraRef = useRef(null);

  // Combined callback for MediaPipe results from both models
  const onFrameResults = useCallback((handsResults, poseResults) => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvasCtx = canvasRef.current.getContext('2d');
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Apply mirroring transformation once for the entire frame
    canvasCtx.translate(canvasRef.current.width, 0);
    canvasCtx.scale(-1, 1);

    // Draw the video frame onto the canvas
    canvasCtx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

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

    // Call onLandmarks with combined data if both models are active
    if (onLandmarks) {
        onLandmarks({
            hands: handsResults?.multiHandLandmarks || null,
            pose: poseResults?.poseLandmarks || null
        });
    }

    canvasCtx.restore();
  }, [onLandmarks]);


  // Load MediaPipe Hands, Pose, Camera_Utils, and Drawing_Utils models
  useEffect(() => {
    let isComponentMounted = true;
    
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

    async function loadMediaPipeLibraries() {
      try {
        if (!isComponentMounted) return;
        
        setIsLoading(true);

        // Check if the global objects are available after loading scripts
        // It's assumed these scripts are loaded in index.html
        if (!window.Hands || !window.Pose || !window.Camera) {
          // If not found, attempt to load them dynamically (fallback, but index.html is preferred)
          await Promise.all([
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js', 'mediapipe-hands-script'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js', 'mediapipe-pose-script'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3/camera_utils.js', 'mediapipe-camera-script'),
            loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3/drawing_utils.js', 'mediapipe-drawing-script')
          ]);
        }
      
        // Check again after potential dynamic load
        if (window.Hands && window.Pose && window.Camera) {
          console.log("Initializing MediaPipe hands and pose models...");
          
          // Only create new instances if they don't exist yet
          if (!handsRef.current) {
            // Initialize MediaPipe Hands 
            const handsInstance = new window.Hands({
              locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
              }
            });

            handsInstance.setOptions({
              maxNumHands: 2,
              modelComplexity: 1,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            });

            // Create a wrapper for onResults that correctly handles our combined approach
            handsInstance.onResults((results) => {
              // Call our combined callback with the hand results and null for pose
              onFrameResults(results, null);
              console.log("Hand model received results");
            });

            handsRef.current = handsInstance;
            setHands(handsInstance);
          }
        
          // Initialize MediaPipe Pose
          if (!poseRef.current) {
            const poseInstance = new window.Pose({
              locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`;
              }
            });

            poseInstance.setOptions({
              modelComplexity: 1,
              smoothLandmarks: true,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            });

            poseInstance.onResults((results) => {
              // Call our combined callback with null for hands and the pose results
              onFrameResults(null, results);
              console.log("Pose model received results");
            });
        
            poseRef.current = poseInstance;
            setPose(poseInstance);
          }
        
          // Set up the Camera instance
          if (videoRef.current && !cameraRef.current) {
            console.log("Setting up camera instance...");
            const cameraInstance = new window.Camera(videoRef.current, {
              onFrame: async () => {
                // Only process if we have active instances and video source
                if (videoRef.current && handsRef.current && poseRef.current) {
                  try {
                    // Process pose first (it's typically less sensitive to hand movements)
                    if (poseRef.current) {
                      await poseRef.current.send({ image: videoRef.current });
                    }

                    // Then process hands
                    if (handsRef.current) {
                      await handsRef.current.send({ image: videoRef.current });
                    }
                  } catch (err) {
                    console.error("Error processing frame with MediaPipe:", err);
                  }
                }
              },
              width: 640,
              height: 480
            });

            cameraRef.current = cameraInstance;
            setCamera(cameraInstance);
            console.log("Camera instance created but not started yet");
          }
        
          if (isComponentMounted) {
            setIsLoading(false);
            console.log("MediaPipe initialization complete");
          }
        } else {
          throw new Error("MediaPipe global objects not found after script load.");
        }
      } catch (err) {
        console.error("Failed to load MediaPipe libraries or initialize model:", err);
        if (isComponentMounted) {
          setError("Failed to load tracking models. Please check your internet connection and try again.");
          setIsLoading(false);
        }
      }
    }

    loadMediaPipeLibraries();

    // Cleanup function
    return () => {
      isComponentMounted = false;
      
      // Clean up camera first
      if (cameraRef.current) {
        console.log("Stopping camera...");
        cameraRef.current.stop();
        setCamera(null);
      }

      // Clean up model instances after a short delay to allow pending operations to complete
      if (stream) {
        console.log("Stopping media stream...");
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }
      
      // Use setTimeout to avoid race conditions
      const cleanupModels = () => {
        if (handsRef.current) {
          console.log("Closing hands model...");
          handsRef.current.close();
          handsRef.current = null;
          setHands(null);
        }
        
        if (poseRef.current) {
          console.log("Closing pose model...");
          poseRef.current.close();
          poseRef.current = null;
          setPose(null);
        }
      };
      
      // Delay model cleanup slightly to avoid issues
      setTimeout(cleanupModels, 300);
    };
  }, []); // Empty dependency array to run only once during component mount

  // Start webcam and MediaPipe camera
  const startWebcam = async () => {
    // Ensure all models and camera are ready before starting
    if (!cameraRef.current || !handsRef.current || !poseRef.current) {
      setError("Tracking models not yet loaded or initialized. Please wait.");
      return;
    }

    try {
      console.log("Starting webcam...");
      // Request media stream with desired resolution
      const s = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });

      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;

        // Important: Make sure to wait until metadata is loaded
        videoRef.current.onloadedmetadata = () => {
          console.log("Video metadata loaded");

          // Set canvas dimensions to match video dimensions once metadata is loaded
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            console.log("Canvas dimensions set:", canvasRef.current.width, canvasRef.current.height);
          }

          // Play the video element
          videoRef.current.play().then(() => {
            console.log("Video playback started");

            // Only start the camera after video is playing
            if (cameraRef.current) {
              console.log("Starting camera processing");
              cameraRef.current.start();
              console.log("Camera started");
            }

            setIsCameraOn(true);
          }).catch(err => {
            console.error("Error playing video:", err);
            setError("Could not play video: " + err.message);
          });
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError("Could not access webcam. Please ensure it's enabled and try again.");
    }
  };

  // Added: Stop webcam and clear output
  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (cameraRef.current) {
      cameraRef.current.stop();
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null; // Clear video source
      videoRef.current.pause(); // Pause video
    }
    // Clear canvas
    if (canvasRef.current) {
      const canvasCtx = canvasRef.current.getContext('2d');
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setHandJsonData('Webcam stopped. No hands detected.');
    setPoseJsonData('Webcam stopped. No pose detected.');
    setIsCameraOn(false); // Updated: Set camera status to false
  };

  // Handle video file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const videoUrl = URL.createObjectURL(file);

      // Stop any active webcam stream before playing file
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        setStream(null);
      }

      if (cameraRef.current) {
        cameraRef.current.stop(); // Stop camera instance as it's for live stream
      }

      if (videoRef.current) {
        videoRef.current.srcObject = null; // Clear previous srcObject
        videoRef.current.src = videoUrl; // Set video file as source
        videoRef.current.load(); // Load the video to ensure metadata is available
      }


      videoRef.current.onloadedmetadata = () => {
        // Set canvas dimensions to match video dimensions
        if (canvasRef.current && videoRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }
        videoRef.current.play();
        setIsCameraOn(true); // Updated: Consider camera "on" when playing a file

        // Process video frames using requestAnimationFrame
        const processVideoFrame = async () => {
          if (videoRef.current && !videoRef.current.paused && !videoRef.current.ended) {
            let currentHandsResults = null;
            let currentPoseResults = null;

            try {
              if (handsRef.current) {
                await handsRef.current.send({ image: videoRef.current });
                // Results will come through the onResults callback
              }
              if (poseRef.current) {
                await poseRef.current.send({ image: videoRef.current });
                // Results will come through the onResults callback
              }
            } catch (err) {
              console.error("Error processing video file frame:", err);
            }
            requestAnimationFrame(processVideoFrame); // Continue loop
          } else if (videoRef.current && videoRef.current.ended) {
            // Clear landmarks when video ends
            setHandJsonData('Video ended. No hands detected.');
            setPoseJsonData('Video ended. No pose detected.');
            setIsCameraOn(false); // Updated: Camera "off" when video ends
          }
        };
        requestAnimationFrame(processVideoFrame);
      };
    }
  };

  return (
    <div className="flex flex-col items-center p-4 bg-gray-100 min-h-screen font-sans w-full">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">MediaPipe Landmark Extractor</h1>

      {isLoading && (
        <div className="text-blue-600 text-lg mb-4">Loading tracking models...</div>
      )}
      {error && (
        <div className="text-red-600 text-lg mb-4 p-3 bg-red-100 border border-red-400 rounded-lg">
          Error: {error}
        </div>
      )}

      <div className="flex flex-col w-full gap-6 max-w-[90%]">
        {/* Video and Canvas Section */}
        <div className="flex-1 flex flex-col items-center bg-white p-4 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-3 text-gray-700">Live Feed with Hands & Pose Tracking</h2>
          <div className="relative w-full max-w-md aspect-video rounded-lg overflow-hidden border-2 border-gray-300 mx-auto">
            {/* The video element is now visible */}
            <video
              ref={videoRef}
              className="absolute top-0 left-0 w-full h-full object-contain bg-black" // Use object-contain to avoid cropping
              autoPlay
              muted
              playsInline
            />
            {/* The canvas displays the video feed with landmarks drawn on it, overlaying the video */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full object-contain" // Use object-contain
            />
          </div>

          <div className="flex flex-wrap justify-center gap-4 mt-6">
            {!isCameraOn ? (
              <button
                onClick={startWebcam}
                disabled={isLoading || !handsRef.current || !poseRef.current || !cameraRef.current}
                className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75 transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Loading...' : 'Enable Webcam'}
              </button>
            ) : (
              <button
                onClick={stopWebcam}
                className="px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-75 transition duration-300 ease-in-out"
              >
                Turn Off Camera
              </button>
            )}

            <label className="px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition duration-300 ease-in-out cursor-pointer">
              Upload Video
              <input
                type="file"
                accept="video/*"
                onChange={handleFileUpload}
                disabled={isLoading}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* JSON Output Section */}
        <div className="flex flex-col md:flex-row gap-6 w-full">
          <div className="flex-1 bg-white p-4 rounded-lg shadow-md flex flex-col">
            <h2 className="text-xl font-semibold mb-3 text-gray-700">Hand Landmarks (JSON Output)</h2>
            <pre className="bg-gray-800 text-green-400 p-4 rounded-lg overflow-auto h-[30vh] text-sm">
              <code>{handJsonData}</code>
            </pre>
          </div>
          <div className="flex-1 bg-white p-4 rounded-lg shadow-md flex flex-col">
            <h2 className="text-xl font-semibold mb-3 text-gray-700">Pose Landmarks (JSON Output)</h2>
            <pre className="bg-gray-800 text-green-400 p-4 rounded-lg overflow-auto h-[30vh] text-sm">
              <code>{poseJsonData}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
