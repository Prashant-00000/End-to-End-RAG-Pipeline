#!/usr/bin/env python3
"""
Deployment Verification Script
Checks if your RAG system is ready for Streamlit Cloud deployment
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist."""
    print("\n📂 Checking files...")
    required_files = [
        "ui.py",
        "main.py",
        "requirements.txt",
        ".gitignore",
        ".streamlit/config.toml",
        ".streamlit/secrets.toml.template",
        "app/__init__.py",
        "app/ingestion.py",
        "app/embedding.py",
        "app/pipeline.py",
        "app/vector_store.py",
        "app/bm25_store.py",
        "app/reranker.py",
        "app/groq_client.py",
        "app/db.py",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ❌ Missing: {file}")
        else:
            print(f"  ✅ {file}")
    
    return len(missing) == 0

def check_environment():
    """Check if environment variables or .env file exists."""
    print("\n🔐 Checking secrets...")
    
    env_path = Path(".env")
    if env_path.exists():
        print("  ✅ .env file found (local)")
    else:
        print("  ⚠️  .env file not found (needed for local dev)")
        print("      → For Streamlit Cloud, add secrets in dashboard")
    
    # Check if secrets in .env
    if env_path.exists():
        with open(".env") as f:
            content = f.read()
        
        required_keys = ["GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
        for key in required_keys:
            if key in content:
                print(f"  ✅ {key} found in .env")
            else:
                print(f"  ⚠️  {key} NOT in .env")
    
    # Check .gitignore excludes secrets
    gitignore_content = Path(".gitignore").read_text()
    if ".env" in gitignore_content and ".streamlit/secrets.toml" in gitignore_content:
        print("  ✅ Secrets properly excluded in .gitignore")
    else:
        print("  ⚠️  Secrets might not be excluded from git")

def check_requirements():
    """Check if requirements.txt has all needed packages."""
    print("\n📦 Checking requirements.txt...")
    
    required_packages = [
        "streamlit",
        "pypdf",
        "sentence-transformers",
        "faiss-cpu",
        "rank-bm25",
        "numpy",
        "groq",
        "python-dotenv",
        "supabase",
    ]
    
    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("  ❌ requirements.txt not found!")
        return False
    
    content = req_path.read_text().lower()
    missing = []
    
    for pkg in required_packages:
        if pkg.lower() in content:
            print(f"  ✅ {pkg}")
        else:
            missing.append(pkg)
            print(f"  ❌ {pkg} MISSING")
    
    return len(missing) == 0

def check_git():
    """Check if git repo is ready."""
    print("\n🔗 Checking Git setup...")
    
    if not Path(".git").exists():
        print("  ⚠️  Not a git repository")
        print("      → Run: git init")
        return False
    else:
        print("  ✅ Git repository initialized")
    
    # Check if remote is set
    try:
        import subprocess
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "origin" in result.stdout:
            print("  ✅ GitHub remote configured")
        else:
            print("  ⚠️  No GitHub remote found")
            print("      → Run: git remote add origin https://github.com/USERNAME/rag-system.git")
    except Exception as e:
        print(f"  ⚠️  Could not check git remote: {e}")
    
    return True

def check_streamlit_config():
    """Check Streamlit configuration."""
    print("\n⚙️  Checking Streamlit config...")
    
    config_path = Path(".streamlit/config.toml")
    if config_path.exists():
        print("  ✅ .streamlit/config.toml exists")
        content = config_path.read_text()
        if "ui.py" not in content:  # Just checking it's properly configured
            if "[theme]" in content:
                print("  ✅ Theme configured")
        if "[server]" in content:
            print("  ✅ Server settings configured")
    else:
        print("  ⚠️  .streamlit/config.toml not found")

def main():
    """Run all checks."""
    print("=" * 60)
    print("🚀 RAG System - Deployment Verification")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # Run checks
    checks = [
        ("Files", check_files),
        ("Secrets", check_environment),
        ("Requirements", check_requirements),
        ("Git", check_git),
        ("Streamlit Config", check_streamlit_config),
    ]
    
    for name, check_func in checks:
        try:
            if check_func():
                checks_passed += 1
        except Exception as e:
            print(f"  ❌ Error during check: {e}")
        checks_total += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Summary: {checks_passed}/{checks_total} checks completed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("\n🎉 Your project is ready for Streamlit Cloud deployment!")
        print("\nNext steps:")
        print("  1. Push to GitHub: git push origin main")
        print("  2. Go to: https://share.streamlit.io/")
        print("  3. Deploy by selecting your repository")
        print("  4. Add secrets in the dashboard")
        print("\n📖 See QUICK_DEPLOY.md for detailed instructions")
    else:
        print("\n⚠️  Please fix the issues above before deploying")
        print("📖 See DEPLOYMENT_GUIDE.md for help")
    
    print()

if __name__ == "__main__":
    main()
