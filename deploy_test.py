#!/usr/bin/env python3
"""
Quick deployment test script
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Run a command and return success/failure"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False

def main():
    print("🚀 Data Analyst Agent - Deployment Test")
    print("=" * 50)
    
    tests = [
        ("docker --version", "Checking Docker installation"),
        ("python --version", "Checking Python installation"),
        ("git status --porcelain", "Checking Git status"),
    ]
    
    for cmd, desc in tests:
        run_command(cmd, desc)
    
    print("\n📋 Deployment Options:")
    print("1. 🐳 Docker: docker build -t data-analyst-agent . && docker run -p 8000:8000 data-analyst-agent")
    print("2. 🐍 Local: python main.py")
    print("3. ☁️ Heroku: git push heroku main")
    print("4. 🌊 Railway: railway up")
    
    print("\n🔗 Your GitHub Repository:")
    print("https://github.com/riyamoun1310/Data-Analyst-Agent")
    
    print("\n✨ Next Steps:")
    print("- Test locally: python main.py")
    print("- Build Docker: docker build -t data-analyst-agent .")
    print("- Deploy to cloud platform of choice")
    print("- Set up monitoring and alerts")

if __name__ == "__main__":
    main()
