const express = require('express');
const path = require('path');
const cors = require('cors');
const translateRoutes = require('./routes/translate');
const learnRoutes = require('./routes/learn');
const fs = require('fs');

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
    console.error(err.stack);
    res.status(500).json({
        error: 'Server error',
        message: err.message
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log(`API endpoints available at:`);
    console.log(`- POST http://localhost:${PORT}/api/translate/live-video`);
    console.log(`- POST http://localhost:${PORT}/api/translate/video`);
    console.log(`- POST http://localhost:${PORT}/api/translate/image`);
});