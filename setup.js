const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    red: '\x1b[31m',
};

// Paths
const clientDir = path.join(__dirname, 'client');
const serverDir = path.join(__dirname, 'server');
const uploadDir = path.join(serverDir, 'uploads');

// Create uploads directory if it doesn't exist
if (!fs.existsSync(uploadDir)) {
    console.log(`${colors.blue}Creating uploads directory...${colors.reset}`);
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Function to run a command in a specific directory
function runCommand(command, cwd) {
    return new Promise((resolve, reject) => {
        console.log(`${colors.bright}${colors.blue}> ${command}${colors.reset}`);
        
        const childProcess = exec(command, { cwd }, (error, stdout, stderr) => {
            if (error) {
                console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
                reject(error);
                return;
            }
            
            if (stderr) {
                console.log(`${colors.yellow}${stderr}${colors.reset}`);
            }
            
            resolve();
        });
        
        childProcess.stdout.on('data', (data) => {
            console.log(data.toString().trim());
        });
    });
}

// Main setup function
async function setup() {
    try {
        console.log(`${colors.bright}${colors.green}=== Setting up VSL-Interpreter ====${colors.reset}`);
        
        // Client setup
        console.log(`\n${colors.bright}${colors.green}=== Installing client dependencies ====${colors.reset}`);
        await runCommand('npm install', clientDir);
        
        // Server setup
        console.log(`\n${colors.bright}${colors.green}=== Installing server dependencies ====${colors.reset}`);
        await runCommand('npm install', serverDir);
        
        // Check for Python
        console.log(`\n${colors.bright}${colors.green}=== Checking Python environment ====${colors.reset}`);
        try {
            await runCommand('python --version', serverDir);
            
            // Install Python dependencies
            console.log(`\n${colors.bright}${colors.green}=== Installing Python dependencies ====${colors.reset}`);
            await runCommand('pip install mediapipe opencv-python numpy', serverDir);
            
            console.log(`\n${colors.bright}${colors.green}=== Setup complete! ====${colors.reset}`);
            console.log(`\n${colors.bright}${colors.blue}To start the server:${colors.reset}`);
            console.log(`   cd server`);
            console.log(`   npm start`);
            console.log(`\n${colors.bright}${colors.blue}To start the client:${colors.reset}`);
            console.log(`   cd client`);
            console.log(`   npm run dev`);
            
        } catch (pythonError) {
            console.error(`${colors.red}Python not found. Please install Python 3.8+ and try again.${colors.reset}`);
            console.error(`${colors.yellow}You need Python for the MediaPipe and ST-GCN++ integration.${colors.reset}`);
        }
        
    } catch (error) {
        console.error(`${colors.red}Setup failed: ${error.message}${colors.reset}`);
    }
}

// Run setup
setup();
