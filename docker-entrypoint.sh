#!/bin/bash
set -e

echo "=================================="
echo "NASA Issue Classification Pipeline"
echo "=================================="
echo ""
echo "Container is starting up..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# Check if .env file exists, if not create from example
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found"
    echo "Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env file with example values"
        echo "  You may need to update API keys for full functionality"
    else
        echo "  No .env.example found, creating minimal .env"
        echo "WANDB_API_KEY=dummy-key" > .env
        echo "OPENAI_API_KEY=dummy-key" >> .env
    fi
    echo ""
fi

# Display configuration info
echo "Configuration:"
if [ -f "config/config.yaml" ]; then
    echo "  Model type: $(grep 'model_type:' config/config.yaml | awk '{print $2}' | tr -d '"')"
    echo "  Random seed: $(grep 'random_seed:' config/config.yaml | awk '{print $2}')"
    echo "  Train data: $(grep 'train_path:' config/config.yaml | awk '{print $2}' | tr -d '"')"
    echo "  Test data: $(grep 'test_path:' config/config.yaml | awk '{print $2}' | tr -d '"')"
else
    echo "  ⚠️  config/config.yaml not found!"
fi
echo ""

echo "Starting application..."
echo "=================================="
echo ""

# Execute the main command
exec "$@"
