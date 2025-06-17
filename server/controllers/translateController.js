const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const workerManager = require('./workerManager');

// Initialize the Python worker when the server starts
(async function initializeWorker() {
    try {
        console.log('Initializing Python inference worker...');
        await workerManager.start();
        console.log('Python inference worker started successfully!');
    } catch (error) {
        console.error('Failed to initialize Python worker:', error);
    }
})();

// Legacy helper function for one-off Python script execution (for file uploads)
function runPythonScript(scriptName, inputData, res) {
    const pythonProcess = spawn('python', [
        path.join(__dirname, '..', scriptName),
        JSON.stringify(inputData),
    ]);

    let result = '';
    let errors = '';

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errors += data.toString();
        console.error(`Python error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0 && result) {
            try {
                const parsedResult = JSON.parse(result);
                res.json({ result: parsedResult });
            } catch (e) {
                console.error('Failed to parse JSON result:', result);
                res.status(500).json({
                    error: 'Failed to parse result',
                    details: result.substring(0, 1000), 
                    pythonErrors: errors
                });
            }
        } else {
            res.status(500).json({
                error: 'Error processing the request',
                code: code,
                pythonErrors: errors,
                stdout: result
            });
        }
    });
}

exports.translateVideo = (req, res) => {
    const videoFile = req.file; 
    if (!videoFile) {
        return res.status(400).json({ error: 'No video file uploaded' });
    }

    const videoPath = videoFile.path;

    // For now, keep using the one-off script for video uploads
    // In the future, this could be adapted to use the worker
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

// Controller for live video translation using the persistent worker
exports.translateLiveVideo = async (req, res) => {
    try {
        // Process using the persistent Python worker
        const result = await workerManager.processData(req.body);
        
        // Send response
        res.json({ result });
    } catch (error) {
        console.error('Error in translateLiveVideo:', error);
        res.status(500).json({
            error: 'Failed to process landmarks',
            message: error.message
        });
    }
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

