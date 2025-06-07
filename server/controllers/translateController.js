const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Helper function to run Python script
function runPythonScript(scriptName, inputData, res) {
    const pythonProcess = spawn('python', [
        path.join(__dirname, '..', scriptName),
        JSON.stringify(inputData),
    ]);

    let result = '';

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.json({ result: JSON.parse(result) });
        } else {
            res.status(500).send('Error processing the request.');
        }
    });
}

exports.translateVideo = (req, res) => {
    const videoFile = req.file; // Assuming you're using multer for file uploads
    if (!videoFile) {
        return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoPath = videoFile.path;

    const pythonProcess = spawn('python', [
        path.join(__dirname, '..', 'stgcn_inference.py'),
        JSON.stringify({ video_path: videoPath }),
    ]);

    let result = '';

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        fs.unlinkSync(videoPath); // Clean up uploaded file
        if (code === 0) {
            res.json(JSON.parse(result));
        } else {
            res.status(500).send('Error processing the video');
        }
    });
};

// Controller for live video translation
exports.translateLiveVideo = (req, res) => {
    const inputData = req.body;
    // Add input type for Python script
    inputData.input_type = "live";
    runPythonScript('stgcn_inference.py', inputData, res);
};

// Controller for pre-recorded video translation
exports.translateVideo = (req, res) => {
    const videoFile = req.file; // Assuming you're using multer for file uploads
    if (!videoFile) {
        return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoPath = videoFile.path;
    const inputData = {
        input_type: "video",
        video_path: videoPath
    };
    
    runPythonScript('stgcn_inference.py', inputData, res);
};

// Controller for image translation
exports.translateImage = (req, res) => {
    const imageFile = req.file; // Assuming you're using multer for file uploads
    if (!imageFile) {
        return res.status(400).json({ error: 'No image file uploaded' });
    }

    const imagePath = imageFile.path;
    const inputData = {
        input_type: "image",
        image_path: imagePath
    };
    
    runPythonScript('stgcn_inference.py', inputData, res);
};