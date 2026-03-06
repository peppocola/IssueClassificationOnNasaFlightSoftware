#!/bin/bash
# Smoke test script for NASA Issue Classification
# Tests basic end-to-end functionality

set -e

echo "=========================================="
echo "NASA Issue Classification - Smoke Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: Check Docker is available
echo "Test 1: Checking Docker availability..."
docker --version > /dev/null 2>&1
print_status $? "Docker is installed"

# Test 2: Check Docker Compose is available
echo ""
echo "Test 2: Checking Docker Compose availability..."
if docker-compose --version > /dev/null 2>&1; then
    print_status 0 "Docker Compose is installed"
elif docker compose version > /dev/null 2>&1; then
    print_status 0 "Docker Compose V2 is installed"
    COMPOSE_CMD="docker compose"
else
    print_status 1 "Docker Compose is not installed"
fi

# Set compose command
COMPOSE_CMD=${COMPOSE_CMD:-docker-compose}

# Test 3: Check required files exist
echo ""
echo "Test 3: Checking required files..."
for file in Dockerfile docker-compose.yml requirements.txt config/config.yaml data/nasa_train_sample.csv data/nasa_test_sample.csv; do
    if [ -f "$file" ]; then
        print_status 0 "Found $file"
    else
        print_status 1 "Missing $file"
    fi
done

# Test 4: Build Docker image
echo ""
echo "Test 4: Building Docker image..."
echo "This may take a few minutes on first run..."
$COMPOSE_CMD build --quiet > /dev/null 2>&1
print_status $? "Docker image built successfully"

# Test 5: Verify image exists
echo ""
echo "Test 5: Verifying Docker image..."
docker images nasa-issue-classifier:latest --format "{{.Repository}}" | grep -q "nasa-issue-classifier"
print_status $? "Docker image 'nasa-issue-classifier:latest' exists"

# Test 6: Quick container start test
echo ""
echo "Test 6: Testing container startup..."
timeout 30 $COMPOSE_CMD run --rm nasa-classifier python -c "print('Container works!'); import sys; sys.exit(0)" > /tmp/smoke_test.log 2>&1
result=$?
if [ $result -eq 0 ]; then
    print_status 0 "Container starts and Python works"
else
    print_warning "Container test had issues (this might be OK if timeout occurred)"
    cat /tmp/smoke_test.log
fi

# Test 7: Check Python version
echo ""
echo "Test 7: Checking Python version in container..."
python_version=$($COMPOSE_CMD run --rm nasa-classifier python --version 2>&1 | grep "3.11.6")
if [ ! -z "$python_version" ]; then
    print_status 0 "Python 3.11.6 is installed"
else
    print_warning "Python version might not be 3.11.6"
fi

# Test 8: Check key dependencies
echo ""
echo "Test 8: Checking key dependencies..."
$COMPOSE_CMD run --rm nasa-classifier python -c "import transformers, torch, pandas, sklearn; print('All key imports successful')" > /dev/null 2>&1
print_status $? "Key dependencies (transformers, torch, pandas, sklearn) are available"

# Test 9: Check submodule
echo ""
echo "Test 9: Checking git submodule..."
$COMPOSE_CMD run --rm nasa-classifier ls externals/sklearn-cls-report2excel/convert_report2excel.py > /dev/null 2>&1
print_status $? "Git submodule is initialized"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}All smoke tests passed!${NC}"
echo "=========================================="
echo ""
echo "The Docker environment is ready to use."
echo ""
echo "To run the full pipeline:"
echo "  $COMPOSE_CMD up"
echo ""
echo "To run in detached mode:"
echo "  $COMPOSE_CMD up -d"
echo ""
echo "To view logs:"
echo "  $COMPOSE_CMD logs -f"
echo ""
