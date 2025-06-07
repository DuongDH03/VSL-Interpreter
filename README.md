# Visual Sign Language Interpreter

A web application that uses MediaPipe and ST-GCN++ for sign language interpretation.

## Features

- Real-time hand and pose detection using MediaPipe
- Sign language translation using ST-GCN++ (Spatio-Temporal Graph Convolutional Network)
- Supports webcam input for live translation
- Accepts video file uploads for translation
- Visualizes hand and pose landmarks in real-time
- API endpoints for different translation needs

## Project Structure

```
client/            # React frontend
  src/
    components/    # React components
      VideoUploader.jsx  # Handles webcam and file input with MediaPipe
    App.jsx        # Main application component
  
server/            # Node.js server
  controllers/     # API controllers
  routes/          # Express routes
  utils/           # Utility modules
    keypoint_extractor.py  # Python module for landmark extraction
  stgcn_inference.py  # Python script for ST-GCN++ inference
  server.js        # Express server setup
```

## Prerequisites

### Frontend
- Node.js 18+
- npm or yarn

### Backend
- Node.js 18+
- Python 3.8+
- MediaPipe
- NumPy
- OpenCV
- (Future) PyTorch for ST-GCN++ model

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd VSL-Interpreter
```

### Install frontend dependencies

```bash
cd client
npm install
```

### Install backend dependencies

```bash
cd server
npm install
pip install mediapipe opencv-python numpy
```

### Start the development server

In one terminal (backend):
```bash
cd server
npm start
# or
node server.js
```

In another terminal (frontend):
```bash
cd client
npm run dev
```

## API Endpoints

### Live Video Translation
- **URL:** `/api/translate/live-video`
- **Method:** `POST`
- **Body:** JSON with hand and pose landmarks from MediaPipe

### Video File Translation
- **URL:** `/api/translate/video`
- **Method:** `POST`
- **Body:** Form data with video file

### Image Translation
- **URL:** `/api/translate/image`
- **Method:** `POST`
- **Body:** Form data with image file

## How It Works

1. **Landmark Detection**:
   - MediaPipe detects hand and pose landmarks from webcam or video input
   - Landmarks are drawn on a canvas for visualization

2. **Keypoint Extraction**:
   - Landmarks are formatted according to the needs of the ST-GCN++ model
   - Buffer collects frames to create fixed-length sequences

3. **Sign Language Recognition**:
   - ST-GCN++ model analyzes spatio-temporal patterns in the landmarks
   - Provides translation based on recognized gestures

4. **User Interface**:
   - Clean React interface for webcam and video input
   - Real-time visualization of detected landmarks
   - Displays translation results

## Future Improvements

- Implement the full ST-GCN++ model for sign language recognition
- Add support for more sign languages
- Improve frame buffering for better recognition
- Add learning mode for users to practice sign language

## License

[MIT License](LICENSE)

## Acknowledgments

- MediaPipe by Google for landmark detection
- ST-GCN++ for sign language recognition architecture
