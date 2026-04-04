#!/bin/bash
# Run the full test suite for TurboQuant
set -e

echo "================================"
echo "TurboQuant Test Suite"
echo "================================"

cd "$(dirname "$0")/.."

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -e ".[dev]"
fi

# Run tests with coverage
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --tb=short -x

echo ""
echo "Running with coverage..."
python -m pytest tests/ --cov=turboquant --cov-report=term-missing --tb=short

echo ""
echo "================================"
echo "All tests passed!"
echo "================================"
