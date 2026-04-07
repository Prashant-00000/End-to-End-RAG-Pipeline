#!/usr/bin/env python3
"""
Deployment Setup Script
Interactive helper to prepare RAG system for Streamlit Cloud deployment
"""

import os
import sys
from pathlib import Path

def create_env_template():
    """Create .env.template if it doesn't exist."""
    template_path = Path(".env.template")
    if not template_path.exists():
        template_path.write_text("""# RAG System Environment Variables
# Copy this file to .env and fill in your values
# For Streamlit Cloud, add these values in Settings → Secrets instead

# Groq API - Get from https://console.groq.com
GROQ_API_KEY=gsk_your_api_key_here
GROQ_TIMEOUT_SECS=120
GROQ_MAX_RETRIES=5

# Supabase - Get from https://supabase.com
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# Optional: Cloud Storage
CLOUD_STORAGE_BUCKET=your-bucket-name
""")
        print("✅ Created .env.template")
    else:
        print("ℹ️  .env.template already exists")

def create_gitignore_entries():
    """Update .gitignore with required entries."""
    gitignore_path = Path(".gitignore")
    required_entries = [
        ".env",
        ".env.local",
        ".streamlit/secrets.toml",
        "__pycache__/",
        "*.pyc",
        ".Python",
        "venv/",
        "ENV/",
    ]
    
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        added = False
        for entry in required_entries:
            if entry not in content:
                gitignore_path.write_text(content + f"\n{entry}")
                print(f"  ✅ Added {entry} to .gitignore")
                added = True
        if not added:
            print("ℹ️  .gitignore already has all required entries")
    else:
        gitignore_path.write_text("\n".join(required_entries))
        print("✅ Created .gitignore")

def init_git():
    """Initialize git repository if needed."""
    git_path = Path(".git")
    if git_path.exists():
        print("ℹ️  Git repository already initialized")
    else:
        os.system("git init")
        print("✅ Initialized git repository")

def setup_github():
    """Help set up GitHub remote."""
    print("\n📝 GitHub Setup")
    print("-" * 40)
    print("Follow these steps:")
    print("1. Create a new repo at: https://github.com/new")
    print("2. Copy the repository URL")
    print("3. Run:")
    print("   git remote add origin <YOUR_URL>")
    print("   git add .")
    print("   git commit -m 'Initial commit'")
    print("   git branch -M main")
    print("   git push -u origin main")

def show_api_setup():
    """Show how to get API keys."""
    print("\n🔑 Get Your API Keys")
    print("-" * 40)
    
    apis = [
        ("Groq API", "https://console.groq.com", "Free tier: $0"),
        ("Supabase", "https://supabase.com", "Free tier: 500GB storage"),
        ("GitHub", "https://github.com", "Free"),
    ]
    
    for name, url, info in apis:
        print(f"\n{name}")
        print(f"  URL: {url}")
        print(f"  {info}")
        print(f"  → Visit and create account")

def generate_deployment_checklist():
    """Generate a deployment checklist."""
    checklist = """
🚀 DEPLOYMENT CHECKLIST
========================

Pre-Deployment:
  ☐ Python 3.8+ installed
  ☐ requirements.txt updated
  ☐ .env file configured (local)
  ☐ GROQ_API_KEY obtained
  ☐ Supabase account created
  ☐ GitHub account ready

Code Setup:
  ☐ Git repository initialized
  ☐ Code committed to GitHub (main branch)
  ☐ .gitignore excludes .env and secrets.toml
  ☐ ui.py is the entry point
  ☐ All imports working locally

GitHub:
  ☐ Repository created on GitHub
  ☐ Code pushed to main branch
  ☐ Repository is PUBLIC (free tier requirement)

Streamlit Cloud Deployment:
  ☐ Streamlit Cloud account (via GitHub)
  ☐ App created and deployed
  ☐ Secrets added in Settings tab (wait 3-5 mins for app to init)

Post-Deployment:
  ☐ App loads without errors
  ☐ Can query LLM
  ☐ Chat history saves

Your App URL:
  https://[YOUR_USERNAME]-rag-system.streamlit.app/
"""
    print(checklist)
    
    # Save to file
    Path("DEPLOYMENT_CHECKLIST.txt").write_text(checklist)
    print("\n✅ Saved to DEPLOYMENT_CHECKLIST.txt")

def main():
    """Main setup flow."""
    print("\n" + "=" * 60)
    print("🚀 RAG System - Deployment Setup")
    print("=" * 60)
    
    print("\n📋 Setting up deployment...")
    
    # 1. Create templates
    print("\n1️⃣  Creating configuration files...")
    create_env_template()
    create_gitignore_entries()
    
    # 2. Git setup
    print("\n2️⃣  Git setup...")
    init_git()
    
    # 3. Show API setup
    show_api_setup()
    
    # 4. Show GitHub steps
    setup_github()
    
    # 5. Generate checklist
    print("\n3️⃣  Generating checklist...")
    generate_deployment_checklist()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\n📖 Next Steps:")
    print("  1. Visit QUICK_DEPLOY.md for 5-minute deployment")
    print("  2. Or read DEPLOYMENT_GUIDE.md for detailed steps")
    print("  3. Get your API keys:")
    print("     • Groq: https://console.groq.com")
    print("     • Supabase: https://supabase.com")
    print("\n📋 Use DEPLOYMENT_CHECKLIST.txt to track progress")
    print()

if __name__ == "__main__":
    main()
