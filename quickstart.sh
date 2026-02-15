#!/bin/bash

# Quick Start Script for SLM Project
# This script automates the setup process

echo "=================================="
echo "🚀 SLM Project Quick Start"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo "   ✅ Virtual environment activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
echo "   This may take 5-10 minutes..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies installed successfully"
else
    echo "   ❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p models
echo "   ✅ Directories created"

# Run data preparation
echo ""
echo "📊 Preparing training data..."
python 1_data_preparation.py

if [ $? -eq 0 ]; then
    echo "   ✅ Training data prepared"
else
    echo "   ❌ Failed to prepare training data"
    exit 1
fi

# Summary
echo ""
echo "=================================="
echo "✨ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. (Optional) Add more training data to data/training_data.json"
echo "2. Start training: python 2_finetune_model.py"
echo "   ⚠️  Training will take 4-8 hours on your hardware"
echo "3. Test your model: python 3_test_model.py"
echo "4. Run chat interface: streamlit run streamlit_app.py"
echo ""
echo "📚 For help, see README.md and TROUBLESHOOTING.md"
echo ""
echo "Happy training! 🎉"
