"""
Model verification script to check if the ST-GCN++ model can be loaded
"""

import os
import sys
import numpy as np

try:
    import torch
    import mmcv
    from mmcv import Config
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install the requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)

try:
    from pyskl.apis import init_recognizer
    from pyskl.datasets.pipelines import Compose
except ImportError:
    print("Error: PYSKL not installed or not in Python path.")
    print("If you cloned PYSKL but haven't installed it:")
    print("cd pyskl")
    print("pip install -e .")
    sys.exit(1)

def check_model_setup():
    """
    Verify that the model configuration and weights can be loaded
    """
    print("Checking model setup...")
    
    # Check if config file exists
    config_file = os.path.join(os.path.dirname(__file__), 'models', 'configs', 'stgcnpp_hand27.py')
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return False
          # Check if weights file exists
    # Look in multiple possible locations
    possible_weight_files = [
        os.path.join(os.path.dirname(__file__), 'models', 'stgcn++_handpose27.pth'),
        os.path.join(os.path.dirname(__file__), 'models', 'stgcn++_model.pth'),
        os.path.join(os.path.dirname(__file__), 'models', 'latest.pth')
    ]
    
    weights_file = None
    for path in possible_weight_files:
        if os.path.exists(path):
            weights_file = path
            break
    
    if weights_file is None:
        print("Error: Model weights not found. Looking for files:")
        for path in possible_weight_files:
            print(f"- {path}")
        print("\nPlease place your ST-GCN++ model weights in the models directory.")
        return False
    else:
        print(f"Found model weights at: {weights_file}")
    
    # Try loading the model
    try:
        # Load configuration
        config = Config.fromfile(config_file)
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize model
        model = init_recognizer(config, weights_file, device)
        model.eval()
        
        # Create a dummy input to test the model
        dummy_input = torch.zeros((1, 3, 64, 27, 1), dtype=torch.float32).to(device)
        
        # Try running inference
        with torch.no_grad():
            output = model(dummy_input, return_loss=False)
            
        print("Model loaded and verified successfully!")
        print(f"Model output shape: {output.shape}")
        
        # Check class map
        class_map_file = os.path.join(os.path.dirname(__file__), 'models', 'class_map.json')
        if os.path.exists(class_map_file):
            import json
            with open(class_map_file, 'r') as f:
                class_map = json.load(f)
            print(f"Found {len(class_map)} classes in class map.")
        else:
            print("Warning: Class map file not found.")
            
        return True
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure that your model weights are compatible with the configuration.")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_model_setup()
    sys.exit(0 if success else 1)
