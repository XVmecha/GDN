#!/bin/bash

# Exit on error
set -e

# Environment name
ENV_NAME="gdn_env"

# Remove environment if it exists
echo "Removing existing environment if it exists..."
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create fresh environment with Python 3.9
echo "Creating new conda environment with Python 3.9..."
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA 10.2
echo "Installing PyTorch..."
conda install pytorch cudatoolkit=10.2 -c pytorch -y

# Install PyG and its dependencies
echo "Installing PyG dependencies..."
conda install pyg -c pyg -y

# If requirements.txt exists, install additional requirements
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements from requirements.txt"
    pip install -r requirements.txt
fi

echo "Installation complete! To use this environment:"
echo "conda activate $ENV_NAME"

# Print versions for verification
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch_geometric; print('PyG version:', torch_geometric.__version__)"