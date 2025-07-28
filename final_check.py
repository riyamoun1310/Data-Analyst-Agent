#!/usr/bin/env python3
"""
Final repository check script
"""

def check_local_files():
    """Check if all required files exist locally"""
    import os
    
    required_files = [
        'main.py',
        'requirements.txt', 
        'README.md',
        'Dockerfile',
        'DEPLOYMENT.md',
        '.env.example',
        'test_api.py'
    ]
    
    print("🔍 Checking local files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    print("🚀 Data Analyst Agent - Final Repository Check")
    print("=" * 50)
    
    if check_local_files():
        print("\n✅ All files present locally!")
        print("\n📋 Manual Steps to Verify GitHub:")
        print("1. 🌐 Visit: https://github.com/riyamoun1310/Data-Analyst-Agent")
        print("2. 🔍 Check if you can see these files:")
        print("   - main.py")
        print("   - README.md") 
        print("   - Dockerfile")
        print("   - requirements.txt")
        print("3. 🔄 If files are missing, the push might not have completed")
        
        print("\n🚀 If repository is empty, try these commands:")
        print("git add .")
        print("git commit -m 'Initial commit with all files'")
        print("git push origin main --force")
        
        print("\n🎯 Your repository URL:")
        print("https://github.com/riyamoun1310/Data-Analyst-Agent")
        
    else:
        print("\n❌ Some files are missing locally!")
        print("Please ensure all files are present before pushing to GitHub")

if __name__ == "__main__":
    main()
