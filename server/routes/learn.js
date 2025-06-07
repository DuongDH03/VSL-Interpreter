const express = require('express');
const { getLessons, submitPractice } = require('../controllers/learnController');

const router = express.Router();

// Get lessons for learning
router.get('/', getLessons);

// Submit practice for evaluation
router.post('/practice', submitPractice);

module.exports = router;