const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { translateLiveVideo, translateVideo, translateImage } = require('../controllers/translateController');

// Configure storage for uploaded files
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '..', 'uploads');
        // Create uploads directory if it doesn't exist
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        // Create unique filename with original extension
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
    fileFilter: (req, file, cb) => {
        // Accept video and image files
        if (file.mimetype.startsWith('video/') || file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Unsupported file format'), false);
        }
    }
});

const router = express.Router();

// Translate sign language from live video (no file upload)
router.post('/live-video', translateLiveVideo);

// Translate sign language from pre-recorded video
router.post('/video', upload.single('video'), translateVideo);


module.exports = router;