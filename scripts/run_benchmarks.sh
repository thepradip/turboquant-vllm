#!/bin/bash
# Run the full benchmark suite
set -e

echo "================================"
echo "TurboQuant Benchmark Suite"
echo "================================"

cd "$(dirname "$0")/.."

echo ""
echo "1. Quality Benchmark"
echo "---"
python -m benchmarks.quality_benchmark

echo ""
echo "2. Memory Benchmark"
echo "---"
python -m benchmarks.memory_benchmark

echo ""
echo "3. Dataset Evaluation"
echo "---"
python -m benchmarks.dataset_eval

echo ""
echo "================================"
echo "All benchmarks complete!"
echo "================================"
