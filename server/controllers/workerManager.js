
const { spawn } = require('child_process');
const path = require('path');

class PythonWorkerManager {
    constructor() {
        this.workerProcess = null;
        this.responseQueue = [];
        this.outputBuffer = '';
        this.isStarting = false;
        this.restartCount = 0;
        this.maxRestarts = 5;
    }

    /**
     * Start the Python worker process
     * @returns {Promise} Resolves when worker is ready
     */
    async start() {
        if (this.workerProcess) {
            console.log('Python worker already running');
            return Promise.resolve();
        }

        if (this.isStarting) {
            console.log('Python worker is already starting');
            // Return a promise that resolves when worker is ready
            return new Promise((resolve, reject) => {
                const checkInterval = setInterval(() => {
                    if (this.workerProcess && !this.isStarting) {
                        clearInterval(checkInterval);
                        resolve();
                    }
                }, 500);
                
                // Timeout after 30 seconds
                setTimeout(() => {
                    clearInterval(checkInterval);
                    reject(new Error('Worker startup timeout'));
                }, 30000);
            });
        }

        this.isStarting = true;

        return new Promise((resolve, reject) => {
            try {
                const workerPath = path.join(__dirname, '..', 'workers', 'pyskl_inference_worker.py');
                console.log(`Starting Python worker: ${workerPath}`);

                // Use python or python3 based on your environment
                this.workerProcess = spawn('python', [workerPath], {
                    stdio: ['pipe', 'pipe', 'pipe'] // stdin, stdout, stderr
                });

                // Set up stdout handler
                this.workerProcess.stdout.on('data', (data) => {
                    this.outputBuffer += data.toString();
                    
                    // Process complete lines
                    let newlineIndex;
                    while ((newlineIndex = this.outputBuffer.indexOf('\n')) !== -1) {
                        const line = this.outputBuffer.substring(0, newlineIndex).trim();
                        this.outputBuffer = this.outputBuffer.substring(newlineIndex + 1);

                        if (line && this.responseQueue.length > 0) {
                            try {
                                const response = JSON.parse(line);
                                const callback = this.responseQueue.shift();
                                callback.resolve(response);
                            } catch (error) {
                                console.error("Error parsing Python output:", error, "Line:", line);
                                if (this.responseQueue.length > 0) {
                                    const callback = this.responseQueue.shift();
                                    callback.reject(new Error("Invalid response from Python worker"));
                                }
                            }
                        }
                    }
                });

                // Set up stderr handler
                this.workerProcess.stderr.on('data', (data) => {
                    console.log(`Python worker [stderr]: ${data.toString().trim()}`);
                    
                    // If stderr contains "Worker initialization complete", worker is ready
                    if (data.toString().includes("Worker initialization complete")) {
                        this.isStarting = false;
                        resolve();
                    }
                });

                // Handle worker exit
                this.workerProcess.on('close', (code) => {
                    console.log(`Python worker exited with code ${code}`);
                    this.workerProcess = null;
                    this.isStarting = false;
                    
                    // Reject all pending requests
                    while (this.responseQueue.length > 0) {
                        const callback = this.responseQueue.shift();
                        callback.reject(new Error("Python worker process closed"));
                    }

                    // Try to restart worker if it crashed unexpectedly
                    if (code !== 0 && this.restartCount < this.maxRestarts) {
                        console.log(`Attempting to restart Python worker (attempt ${this.restartCount + 1}/${this.maxRestarts})...`);
                        this.restartCount++;
                        setTimeout(() => this.start(), 5000);
                    } else if (this.restartCount >= this.maxRestarts) {
                        console.error(`Maximum restart attempts (${this.maxRestarts}) reached. Not restarting worker.`);
                    }
                });

                // Handle spawn errors
                this.workerProcess.on('error', (error) => {
                    console.error(`Failed to start Python worker: ${error.message}`);
                    this.workerProcess = null;
                    this.isStarting = false;
                    reject(error);
                });

                // Set timeout for startup
                setTimeout(() => {
                    if (this.isStarting) {
                        this.isStarting = false;
                        console.log("Worker startup timed out, but process may still be initializing");
                        resolve(); // Resolve anyway and hope it eventually starts
                    }
                }, 30000);
                
                // Send a ping after a short delay to check if worker is responsive
                setTimeout(() => {
                    this.ping().then(() => {
                        this.isStarting = false;
                        resolve();
                    }).catch(err => {
                        console.log("Worker ping failed during startup:", err);
                        // Don't reject, as stderr might still indicate success
                    });
                }, 5000);
                
            } catch (error) {
                console.error(`Error starting Python worker: ${error.message}`);
                this.isStarting = false;
                reject(error);
            }
        });
    }

    /**
     * Stop the Python worker process
     */
    stop() {
        if (this.workerProcess) {
            console.log('Stopping Python worker...');
            this.workerProcess.kill();
            this.workerProcess = null;
        }
    }

    /**
     * Send a request to the Python worker and get a response
     * @param {Object} data - Data to send to the worker
     * @returns {Promise} - Resolves with the worker's response
     */
    async sendRequest(data) {
        if (!this.workerProcess) {
            await this.start();
        }

        return new Promise((resolve, reject) => {
            // Add to response queue
            this.responseQueue.push({ resolve, reject });
            
            // Send data to Python worker
            this.workerProcess.stdin.write(JSON.stringify(data) + '\n');
        });
    }

    /**
     * Process landmarks using the Python worker
     * @param {Array} handLandmarks - Hand landmarks data
     * @param {Array} poseLandmarks - Pose landmarks data
     * @returns {Promise} - Resolves with inference results
     */
    async processLandmarks(handLandmarks, poseLandmarks) {
        return this.sendRequest({
            type: 'landmarks',
            hands: handLandmarks,
            pose: poseLandmarks
        });
    }

    /**
     * Reset the worker's internal buffer
     * @returns {Promise} - Resolves when buffer is reset
     */
    async resetBuffer() {
        return this.sendRequest({ type: 'reset' });
    }

    /**
     * Ping the worker to check if it's alive
     * @returns {Promise} - Resolves if worker responds
     */
    async ping() {
        return this.sendRequest({ type: 'ping' });
    }
}

// Create and export a singleton instance
const workerManager = new PythonWorkerManager();

module.exports = workerManager;
