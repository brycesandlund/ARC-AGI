#!/bin/bash
# TPU Setup Script for ARC-AGI Project

# Install required packages on TPU VM
setup_tpu_vm() {
    echo "Installing required packages..."
    pip install --upgrade pip
    
    # Install PyTorch/XLA and other dependencies
    pip install torch==2.0.0 torch_xla==2.0.0 "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install transformers datasets trl wandb
    
    # Check TPU connectivity
    python3 -c "import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())"
    
    echo "TPU setup complete!"
}

# Run the training script
run_training() {
    echo "Starting training..."
    python SFT_arc_tpu.py
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
