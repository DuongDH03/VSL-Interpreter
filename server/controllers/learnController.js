exports.getLessons = (req, res) => {
    // Return a list of lessons (mock data for now)
    res.json({
        lessons: [
            // holder for actual lesson :/
            { id: 1, name: 'Lesson 1: Basic Gestures' },
            { id: 2, name: 'Lesson 2: Common Words' },
        ],
    });
};

exports.submitPractice = (req, res) => {
    const practiceData = req.body;
    // Evaluate the user's practice (mock response for now)
    res.json({ message: 'Practice submitted successfully', score: 85 });
};