#!/bin/bash
# Startup script for Scout backend server

cd "$(dirname "$0")"

# Use existing virtual environment from ../../../.venv
if [ ! -d "../../../.venv" ]; then
    echo "Error: Virtual environment not found at ../../../.venv"
    exit 1
fi

echo "Activating virtual environment from ../../../.venv..."
source ../../../.venv/bin/activate

# Install dependencies if needed
echo "Ensuring dependencies are installed..."
pip install -q flask flask-cors pytest requests

# Start server
echo "Starting Flask server on http://localhost:5000..."
python server.py
