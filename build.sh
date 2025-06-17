# Exit on error
set -e

# Log commands
set -x

# Install specific Python version if not available
if ! command -v python3.8 &> /dev/null; then
    echo "Installing Python 3.8..."
    apt-get update
    apt-get install -y python3.8 python3.8-dev python3.8-distutils
    ln -sf /usr/bin/python3.8 /usr/bin/python3
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.8 get-pip.py
    rm get-pip.py
fi

# Install server Node.js dependencies
cd server
npm ci

# Install Python requirements
python3.8 -m pip install --upgrade pip
python3.8 -m pip install -r requirements.txt
cd ..

# Make the build script executable
chmod +x build.sh

# Print versions for debugging
python3.8 --version
node --version
npm --version