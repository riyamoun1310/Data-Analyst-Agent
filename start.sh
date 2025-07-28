#!/bin/bash
# Render.com startup script

echo "🚀 Starting Data Analyst Agent on Render..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start the application
echo "🌟 Starting FastAPI server..."
python main.py
