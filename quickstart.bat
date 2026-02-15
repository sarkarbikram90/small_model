@echo off
REM Quick Start Script for SLM Project (Windows)

echo ==================================
echo 🚀 SLM Project Quick Start
echo ==================================
echo.

REM Check Python version
echo 📋 Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)
echo    ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo    ✅ Virtual environment created
) else (
    echo    ℹ️  Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat
echo    ✅ Virtual environment activated
echo.

REM Install dependencies
echo 📥 Installing dependencies...
echo    This may take 5-10 minutes...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo    ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo    ✅ Dependencies installed successfully
echo.

REM Create necessary directories
echo 📁 Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
echo    ✅ Directories created
echo.

REM Run data preparation
echo 📊 Preparing training data...
python 1_data_preparation.py

if errorlevel 1 (
    echo    ❌ Failed to prepare training data
    pause
    exit /b 1
)
echo    ✅ Training data prepared
echo.

REM Summary
echo ==================================
echo ✨ Setup Complete!
echo ==================================
echo.
echo Next steps:
echo 1. (Optional) Add more training data to data\training_data.json
echo 2. Start training: python 2_finetune_model.py
echo    ⚠️  Training will take 4-8 hours on your hardware
echo 3. Test your model: python 3_test_model.py
echo 4. Run chat interface: streamlit run streamlit_app.py
echo.
echo 📚 For help, see README.md and TROUBLESHOOTING.md
echo.
echo Happy training! 🎉
echo.
pause
