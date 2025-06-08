"""
Simple script to install directly from requirements.txt
and clone PYSKL repository for manual installation
"""

import subprocess
import sys
import os
import platform

def install_from_requirements():
    """
    Install packages from requirements.txt
    """
    print("Installing packages from requirements.txt...")
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print(f"Error: requirements.txt not found at {requirements_path}")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Successfully installed packages from requirements.txt")
    except subprocess.CalledProcessError:
        print("Failed to install packages from requirements.txt")
        return False
    
    print("\nCloning PYSKL repository...")
    try:
        # Create a directory for the clone
        pyskl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyskl')
        os.makedirs(pyskl_dir, exist_ok=True)
        
        # Clone the repository
        clone_cmd = f"git clone https://github.com/kennymckormick/pyskl.git {pyskl_dir}"
        subprocess.check_call(clone_cmd, shell=True)
        
        print(f"\nPYSKL repository cloned to {pyskl_dir}")
        print("\nTo install PYSKL, run the following commands:")
        print(f"cd {pyskl_dir}")
        print("pip install -e .")
        
        return True
    except Exception as e:
        print(f"Error cloning PYSKL repository: {e}")
        return False

if __name__ == "__main__":
    success = install_from_requirements()
    sys.exit(0 if success else 1)
