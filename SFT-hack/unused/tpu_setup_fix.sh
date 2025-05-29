#!/bin/bash
# TPU Setup Script for ARC-AGI Project

# Check Python version
echo "Python version:"
python --version

# Install required packages on TPU VM
setup_tpu_vm() {
    echo "Installing required packages..."
    pip install --upgrade pip
    
    # Fix package dependencies first
    pip install packaging>=20.9 requests>=2.32.2 typing-extensions>=4.12.2
    
    # Install PyTorch/XLA and other dependencies
    pip install torch==2.0.0 torch_xla==2.0.0 "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install transformers datasets trl wandb
    
    # Add local bin to PATH
    export PATH=$HOME/.local/bin:$PATH
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    
    # Check TPU connectivity
    python -c "import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())"
    
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
