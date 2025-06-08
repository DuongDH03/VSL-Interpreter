"""
Script to install the required packages for integration with existing PYSKL model

This script will install the necessary dependencies for the ST-GCN++ model:
- PyTorch 1.12.1 (compatible with Python 3.8)
- MMCV-full 1.5.0
- MMDet 2.23.0
- MMPose 0.24.0
- PYSKL (installed from source)

Recommended: Python 3.8
"""

import subprocess
import sys
import os
import platform

def install_requirements():
    """
    Install required packages for the ST-GCN++ model
    """
    print("Setting up environment for ST-GCN++ model...")
    
    # Check Python version
    python_version = platform.python_version()
    python_major, python_minor, _ = map(int, python_version.split('.'))
    print(f"Python version: {python_version}")
    
    # Check if Python version is compatible
    if python_major != 3 or python_minor != 8:
        print(f"Warning: This setup script is optimized for Python 3.8.")
        print(f"You're using Python {python_major}.{python_minor}.")
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            print("Setup cancelled.")
            return
      # Required base packages
    basic_packages = [
        "numpy>=1.19.5",
        "opencv-python",
        "opencv-contrib-python",
        "mediapipe",
        "decord>=0.6.0",
        "fvcore",
        "matplotlib",
        "moviepy",
        "pymemcache",
        "scipy",
        "tqdm"
    ]    # PyTorch installation - CPU-only version compatible with Python 3.8
    torch_cmd = "pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu"
    
    # MMCV installation - specific version for compatibility
    mmcv_cmd = "pip install mmcv-full==1.5.0"
    
    # MM-series packages at specific versions
    mm_packages = [
        "mmdet==2.23.0",
        "mmpose==0.24.0"
    ]
    
    print("Installing basic packages...")
    for package in basic_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    print("\nInstalling PyTorch...")
    try:
        subprocess.check_call(torch_cmd.split())
        print("Successfully installed PyTorch")
    except subprocess.CalledProcessError:
        print("Failed to install PyTorch")
        print("\nInstalling MMCV (this may take a while)...")
    try:
        subprocess.check_call(mmcv_cmd.split())
        print("Successfully installed MMCV")
    except subprocess.CalledProcessError:
        print("Failed to install MMCV")
    
    print("\nInstalling MM-series packages...")
    for package in mm_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    print("\nCloning and installing PYSKL from source...")
    try:
        # Create a temporary directory for the clone
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Clone the repository
        clone_cmd = f"git clone https://github.com/kennymckormick/pyskl.git {temp_dir}"
        subprocess.check_call(clone_cmd, shell=True)
        
        # Change to the directory and install in development mode
        cwd = os.getcwd()
        os.chdir(temp_dir)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        os.chdir(cwd)
        
        print("Successfully installed PYSKL")
    except Exception as e:
        print(f"Failed to install PYSKL: {e}")
    
    print("\nSetup complete!")
    print("You can now use the ST-GCN++ model for sign language recognition.")
    
if __name__ == "__main__":
    install_requirements()
