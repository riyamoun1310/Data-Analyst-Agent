#!/usr/bin/env python3
"""
Repository verification script
"""

import requests
import json

def check_github_repo():
    """Check if the GitHub repository is accessible"""
    repo_url = "https://api.github.com/repos/riyamoun1310/Data-Analyst-Agent"
    
    try:
        print("🔍 Checking GitHub repository...")
        response = requests.get(repo_url)
        
        if response.status_code == 200:
            repo_data = response.json()
            print("✅ Repository is LIVE and accessible!")
            print(f"📊 Repository Details:")
            print(f"   - Name: {repo_data['name']}")
            print(f"   - Description: {repo_data.get('description', 'No description')}")
            print(f"   - Stars: {repo_data['stargazers_count']}")
            print(f"   - Language: {repo_data.get('language', 'Not detected')}")
            print(f"   - Size: {repo_data['size']} KB")
            print(f"   - Last updated: {repo_data['updated_at']}")
            print(f"   - Clone URL: {repo_data['clone_url']}")
            
            # Check if main files exist
            files_to_check = ['main.py', 'requirements.txt', 'README.md', 'Dockerfile']
            print(f"\n📁 Checking essential files...")
            
            for file in files_to_check:
                file_url = f"https://api.github.com/repos/riyamoun1310/Data-Analyst-Agent/contents/{file}"
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    print(f"   ✅ {file} - Found")
                else:
                    print(f"   ❌ {file} - Missing")
            
            return True
        else:
            print(f"❌ Repository not accessible. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking repository: {str(e)}")
        return False

def main():
    print("🚀 Data Analyst Agent - Repository Verification")
    print("=" * 50)
    
    if check_github_repo():
        print("\n🎉 SUCCESS! Your repository is live at:")
        print("🔗 https://github.com/riyamoun1310/Data-Analyst-Agent")
        
        print("\n📋 Next Steps:")
        print("1. 🌟 Star your own repository")
        print("2. 📝 Update README.md if needed")
        print("3. 🚀 Deploy to your preferred platform:")
        print("   - Heroku: Connect GitHub repo")
        print("   - Railway: Import from GitHub")
        print("   - Vercel: Import project")
        print("   - Google Cloud Run: Use GitHub integration")
        
        print("\n🧪 Test your API locally:")
        print("   python main.py")
        print("   curl http://localhost:8000/health")
        
    else:
        print("\n❌ Repository verification failed")
        print("Please check your GitHub repository manually")

if __name__ == "__main__":
    main()
