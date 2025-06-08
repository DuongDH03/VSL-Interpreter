const { spawn } = require('child_process');

// Function to check Python version and installed packages
function checkPythonEnvironment() {
    console.log('Checking Python environment...');

    // Check Python version
    const pythonVersion = spawn('python', ['--version']);
    pythonVersion.stdout.on('data', (data) => {
        console.log(`Python version: ${data}`);
    });
    pythonVersion.stderr.on('data', (data) => {
        console.log(`Python version: ${data}`);
    });

    // List installed packages
    const pipList = spawn('python', ['-m', 'pip', 'list']);
    let packages = '';
    pipList.stdout.on('data', (data) => {
        packages += data.toString();
    });
    pipList.on('close', () => {
        console.log('Installed Python packages:');
        
        // Check for required packages
        const requiredPackages = ['numpy', 'torch', 'opencv-python', 'mediapipe'];
        for (const pkg of requiredPackages) {
            if (packages.includes(pkg)) {
                console.log(`✓ ${pkg} is installed`);
            } else {
                console.log(`✗ ${pkg} is NOT installed`);
            }
        }
        
        // Check if PYSKL is installed
        if (packages.includes('pyskl')) {
            console.log('✓ PYSKL is installed');
        } else {
            console.log('✗ PYSKL is NOT installed - you may need to run setup_pyskl.py');
        }
    });
}

// Run check
checkPythonEnvironment();

module.exports = { checkPythonEnvironment };
