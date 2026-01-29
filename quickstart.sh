#!/bin/bash
# Quick start script for 3DMoEST

echo "==================================="
echo "3DMoEST Quick Start"
echo "==================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA availability
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✓ CUDA is available"
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "  GPU count: $gpu_count"
else
    echo "⚠ CUDA not available - training will be slow"
fi

echo ""
echo "==================================="
echo "Installation Steps:"
echo "==================================="
echo "1. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Download UNI model:"
echo "   Follow: https://github.com/mahmoodlab/UNI"
echo "   Place in: ./models/UNI/"
echo ""
echo "3. Prepare your data:"
echo "   - Format: H5 file with patches, expression, coords_3d"
echo "   - Run: python src/data_preparation/step2_extract_*.py"
echo ""
echo "4. Train model:"
echo "   python src/training/train_moest_plus_final.py"
echo ""
echo "5. Run inference:"
echo "   python src/inference/inference_her2_moest.py"
echo ""
echo "==================================="
echo "For detailed instructions, see:"
echo "  - README.md"
echo "  - docs/CLAUDE.md"
echo "==================================="
