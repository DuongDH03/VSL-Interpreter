
import sys
import os
import importlib.util

def check_module(name):
    """Check if a module is available"""
    spec = importlib.util.find_spec(name)
    return spec is not None

def check_pyskl_environment():
    """Check if the environment has all necessary components for PySKL"""
    print("Checking PySKL environment...")
    
    # Check Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    if not python_version.startswith("3.8"):
        print("⚠️ Warning: Recommended Python version is 3.8")
    
    # Check PyTorch
    if check_module('torch'):
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ PyTorch is not installed")
    
    # Check MMCV
    if check_module('mmcv'):
        import mmcv
        print(f"✅ MMCV version: {mmcv.__version__}")
    else:
        print("❌ MMCV is not installed")
    
    # Check MM-series packages
    mm_packages = ['mmdet', 'mmpose']
    for pkg in mm_packages:
        if check_module(pkg):
            module = importlib.import_module(pkg)
            print(f"✅ {pkg.capitalize()} version: {module.__version__ if hasattr(module, '__version__') else 'unknown'}")
        else:
            print(f"❌ {pkg.capitalize()} is not installed")
    
    # Check for PySKL
    if check_module('pyskl'):
        print("✅ PySKL is installed")
    else:
        print("❌ PySKL is not installed")
        
        # Check common locations for PySKL repo
        server_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(server_dir, 'pyskl'),
            os.path.join(os.path.dirname(server_dir), 'pyskl'),
            os.path.expanduser('~/pyskl')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   Found PySKL repository at: {path}")
                if os.path.join(path, 'pyskl', '__init__.py') not in {None, ''}:
                    print(f"   You can add this to PYTHONPATH: {path}")
                    break
    
    # Check for model files
    server_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(server_dir, 'models', 'configs', 'stgcnpp_hand27.py')
    model_file = os.path.join(server_dir, 'models', 'stgcn++_model.pth')
    
    if os.path.exists(config_file):
        print(f"✅ Model config found: {config_file}")
    else:
        print(f"❌ Model config not found: {config_file}")
    
    if os.path.exists(model_file):
        print(f"✅ Model weights found: {model_file}")
        print(f"   Size: {os.path.getsize(model_file) / (1024*1024):.2f} MB")
    else:
        print(f"❌ Model weights not found: {model_file}")
    
if __name__ == "__main__":
    check_pyskl_environment()
