#!/bin/bash
# Improved TPU Setup Script for ARC-AGI Project

# Check Python version and environment
echo "Python version:"
python --version
echo "Python path:"
which python

# Install required packages on TPU VM
setup_tpu_vm() {
    echo "Installing required packages..."
    
    # TPU VMs typically come with PyTorch/XLA pre-installed
    # Let's verify what's available
    echo "Checking for pre-installed PyTorch/XLA:"
    python -c "import torch; print('PyTorch version:', torch.__version__)" || echo "PyTorch not found"
    python -c "import torch_xla; print('PyTorch/XLA available')" || echo "PyTorch/XLA not found"
    
    # If PyTorch/XLA is not available, use TPU VM's recommended installation
    if ! python -c "import torch_xla" &> /dev/null; then
        echo "Installing PyTorch/XLA using TPU VM's recommended approach..."
        # This should work for most TPU VM images
        pip install --upgrade pip
        
        # For TPU VMs, the correct PyTorch/XLA is often provided through special URLs
        pip install torch torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html
    else
        echo "Using pre-installed PyTorch/XLA"
    fi
    
    # Fix package dependencies
    pip install packaging>=20.9 requests>=2.32.2 typing-extensions>=4.12.2 --upgrade
    
    # Install other dependencies
    pip install transformers datasets trl wandb --upgrade
    
    # Add local bin to PATH
    export PATH=$HOME/.local/bin:$PATH
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    
    # Check TPU connectivity
    python -c "import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())" || echo "Failed to access TPU device"
    
    echo "TPU setup complete!"
}

# Run the training script
run_training() {
    echo "Starting training..."
    # Use the fixed script without f-strings
    python SFT_arc_tpu_fix.py
}

# Export TPU-related environment variables
export TPU_ACCELERATOR_TYPE=$(curl -s -f -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" || echo "unknown")
export TPU_NUM_DEVICES=8  # For v3-8

# Create directory for checkpoints
mkdir -p ./tpu_trained_model

# Execute setup
setup_tpu_vm

# Run training
run_training
