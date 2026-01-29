#!/bin/bash
# Quick start script for Spateo UNI alignment

echo "=========================================="
echo "Spateo UNI Alignment - Quick Start"
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import numpy, scipy, h5py, torch, scanpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Core dependencies installed"
else
    echo "✗ Missing dependencies. Installing..."
    pip install numpy scipy h5py torch scanpy anndata matplotlib seaborn
fi

# Check for Spateo
python -c "import spateo" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Spateo installed"
else
    echo "⚠ Spateo not installed (optional)"
    echo "  Install with: pip install spateo-release"
fi

echo ""
echo "=========================================="
echo "Usage Examples:"
echo "=========================================="
echo ""
echo "1. Basic alignment (all slices):"
echo "   python src/alignment/spateo_uni_alignment.py"
echo ""
echo "2. Run example 1 (basic):"
echo "   python src/alignment/example_usage.py --example 1"
echo ""
echo "3. Run example 2 (UNI-only):"
echo "   python src/alignment/example_usage.py --example 2"
echo ""
echo "4. Run example 3 (selective slices):"
echo "   python src/alignment/example_usage.py --example 3"
echo ""
echo "5. Run example 4 (parameter comparison):"
echo "   python src/alignment/example_usage.py --example 4"
echo ""
echo "6. Run example 5 (downstream analysis):"
echo "   python src/alignment/example_usage.py --example 5"
echo ""
echo "7. Run all examples:"
echo "   python src/alignment/example_usage.py --all"
echo ""
echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo ""
echo "Set environment variables:"
echo "  export DATA_PATH='./data/dataset.h5'"
echo "  export OUTPUT_DIR='./results/alignment'"
echo ""
echo "Or edit configuration in the script:"
echo "  src/alignment/spateo_uni_alignment.py"
echo ""
echo "=========================================="
