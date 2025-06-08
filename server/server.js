const express = require('express');
const path = require('path');
const cors = require('cors');
const translateRoutes = require('./routes/translate');
const learnRoutes = require('./routes/learn');
const fs = require('fs');
const { checkPythonEnvironment } = require('./check_environment');

const app = express();
const PORT = 3001;

// Enable CORS for development (can be restricted in production)
app.use(cors());

// Parse JSON and URL-encoded requests
app.use(express.json({ limit: '50mb' })); // Increase limit for larger JSON payloads
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// API Routes
app.use('/api/translate', translateRoutes);
app.use('/api/learn', learnRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err.message);
    console.error(err.stack);
    res.status(500).json({
        error: 'Server error',
        message: err.message,
        stack: process.env.NODE_ENV === 'production' ? undefined : err.stack
    });
});

// Import worker manager 
const workerManager = require('./controllers/workerManager');

// Start the server
app.listen(PORT, async () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log(`API endpoints available at:`);
    console.log(`- POST http://localhost:${PORT}/api/translate/live-video`);
    console.log(`- POST http://localhost:${PORT}/api/translate/video`);
    console.log(`- POST http://localhost:${PORT}/api/translate/image`);
    
    // Check Python environment on startup
    checkPythonEnvironment();
    
    // Start Python inference worker
    try {
        console.log('Starting Python inference worker...');
        await workerManager.start();
        console.log('Python inference worker started successfully!');
    } catch (error) {
        console.error('Failed to start Python worker:', error.message);
        console.log('Server will continue running, but real-time translation may not work properly');
    }
});

// Handle graceful shutdown
process.on('SIGINT', async () => {
    console.log('Shutting down server...');
    // Stop the Python worker
    workerManager.stop();
    process.exit(0);
});