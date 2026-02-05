#!/bin/bash
# Docker Validation Script
# This script validates the Docker setup for the NASA Issue Classification project

set -e

echo "=== Docker Setup Validation ==="
echo ""

# Check if Docker is available
echo "1. Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi
echo "✓ Docker is available"
echo ""

# Check if docker-compose is available
echo "2. Checking Docker Compose availability..."
if ! command -v docker-compose &> /dev/null; then
    echo "WARNING: docker-compose is not installed (optional)"
else
    echo "✓ Docker Compose is available"
fi
echo ""

# Build the Docker image
echo "3. Building Docker image..."
docker build -t nasa-issue-classifier:validation . > /dev/null 2>&1
echo "✓ Docker image built successfully"
echo ""

# Test Python version
echo "4. Checking Python version in container..."
PYTHON_VERSION=$(docker run --rm nasa-issue-classifier:validation python --version)
if [[ $PYTHON_VERSION == *"3.11.6"* ]]; then
    echo "✓ Python version: $PYTHON_VERSION"
else
    echo "ERROR: Expected Python 3.11.6, got $PYTHON_VERSION"
    exit 1
fi
echo ""

# Test dependency versions
echo "5. Checking dependency versions..."
docker run --rm nasa-issue-classifier:validation python -c "
import transformers, torch, numpy, pandas, sklearn
assert transformers.__version__ == '4.39.0', f'transformers version mismatch: {transformers.__version__}'
assert torch.__version__.startswith('2.2.2'), f'torch version mismatch: {torch.__version__}'
assert numpy.__version__ == '1.26.4', f'numpy version mismatch: {numpy.__version__}'
assert pandas.__version__ == '2.2.2', f'pandas version mismatch: {pandas.__version__}'
assert sklearn.__version__ == '1.4.2', f'sklearn version mismatch: {sklearn.__version__}'
print('✓ All dependency versions match')
" 2>&1 | tail -1
echo ""

# Test config loading
echo "6. Testing config loading..."
docker run --rm \
    -v $(pwd)/config:/app/config:ro \
    nasa-issue-classifier:validation \
    python -c "from config.config_loader import load_config; config = load_config('config/config.yaml'); print('✓ Config loaded successfully')" 2>&1 | tail -1
echo ""

# Test data processing imports
echo "7. Testing data processing imports..."
docker run --rm nasa-issue-classifier:validation \
    python -c "from data_processing.dataset_utils import preprocess_dataset, print_label_distribution; print('✓ Data processing imports work')" 2>&1 | tail -1
echo ""

# Clean up
echo "8. Cleaning up..."
docker image rm nasa-issue-classifier:validation > /dev/null 2>&1
echo "✓ Cleanup complete"
echo ""

echo "=== All validation checks passed! ==="
echo ""
echo "Your Docker setup is ready. To run the full pipeline:"
echo "  docker-compose build"
echo "  docker-compose up"
