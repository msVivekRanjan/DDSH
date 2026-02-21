#!/bin/bash
# setup.sh — Complete setup script for DDSH project
# Run this once to set up the entire environment

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  DDSH Setup — Driver Drowsiness Shield             ║"
echo "║  Automated Environment Configuration              ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Check Python version
echo -e "${BLUE}[Step 1]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Install Python 3.10+ and try again.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Step 2: Create virtual environment
echo ""
echo -e "${BLUE}[Step 2]${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 3: Activate virtual environment
echo ""
echo -e "${BLUE}[Step 3]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 4: Upgrade pip
echo ""
echo -e "${BLUE}[Step 4]${NC} Upgrading pip..."
pip install --quiet --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 5: Install dependencies
echo ""
echo -e "${BLUE}[Step 5]${NC} Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 6: Download Haar cascades
echo ""
echo -e "${BLUE}[Step 6]${NC} Downloading Haar cascade classifiers..."
cd scripts
python download_haarcascades.py
cd ..
echo -e "${GREEN}✓ Haar cascades ready${NC}"

# Step 7: Verify dataset structure
echo ""
echo -e "${BLUE}[Step 7]${NC} Checking dataset structure..."
if [ -d "data/train/Open_Eyes" ] && [ -d "data/train/Closed_Eyes" ]; then
    OPEN_COUNT=$(find data/train/Open_Eyes -type f | wc -l)
    CLOSED_COUNT=$(find data/train/Closed_Eyes -type f | wc -l)
    echo -e "${GREEN}✓ Training data found:${NC}"
    echo "   - Open Eyes: $OPEN_COUNT images"
    echo "   - Closed Eyes: $CLOSED_COUNT images"
else
    echo -e "${RED}⚠️  Training data not found at data/train/${NC}"
    echo "   Download MRL Eyes 2018 dataset: http://mrl.cs.vsb.cz/eyedataset"
    echo "   Extract to: data/train/ and data/test/"
fi

# Step 8: Final verification
echo ""
echo -e "${BLUE}[Step 8]${NC} Final verification..."
python3 -c "import tensorflow, cv2, sklearn; print('✓ All imports successful')"
echo -e "${GREEN}✓ Environment verified${NC}"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download dataset: http://mrl.cs.vsb.cz/eyedataset"
echo "  3. Train model: cd scripts && python train.py"
echo "  4. Evaluate: python evaluate.py"
echo "  5. Run detection: python detect.py"
echo ""
