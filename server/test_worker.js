const workerManager = require('./controllers/workerManager');

// Sample hand and pose data for testing
const sampleData = {
    hands: [
        // 21 landmarks for one hand with x, y values
        Array(21).fill().map(() => ({
            x: Math.random(),
            y: Math.random(),
            z: Math.random()
        }))
    ],
    pose: Array(33).fill().map(() => ({
        x: Math.random(),
        y: Math.random(),
        visibility: Math.random()
    }))
};

async function testWorker() {
    try {
        console.log('Starting Python inference worker...');
        await workerManager.start();
        console.log('Worker started successfully!');

        console.log('Sending ping request...');
        const pingResult = await workerManager.ping();
        console.log('Ping result:', pingResult);

        console.log('Processing sample landmarks...');
        const result = await workerManager.processLandmarks(sampleData.hands, sampleData.pose);
        console.log('Inference result:', JSON.stringify(result, null, 2));

        console.log('Resetting buffer...');
        const resetResult = await workerManager.resetBuffer();
        console.log('Reset result:', resetResult);

        console.log('Stopping worker...');
        workerManager.stop();
        console.log('Worker stopped');
    } catch (error) {
        console.error('Error in test:', error);
    } finally {
        process.exit(0);
    }
}

testWorker();
