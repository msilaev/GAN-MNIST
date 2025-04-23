#!/bin/bash

# Exit on error
#set -e

# Optional: Activate virtual environment
# source /path/to/venv/bin/activate

# Inform the user
echo "Starting conditional GAN training..."

# Run the training script
python3 train_conditional_gan.py

# Confirm completion
echo "Training complete!"
