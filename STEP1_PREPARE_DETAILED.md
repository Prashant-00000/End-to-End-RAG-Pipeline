# 🎯 Step 1: Prepare - DETAILED EXPLANATION

## What is "Prepare"?

Before pushing your code to GitHub and deploying, you need to make sure everything is set up correctly. This step checks and prepares your project.

---

## 🔍 Option A: VERIFY Your Project is Ready

### What it does:
Checks if your project has all the files and configurations needed for cloud deployment.

### How to run:
```bash
python verify_deployment.py
```

### What happens:
The script will check:
- ✅ All Python files exist (`ui.py`, `main.py`, etc.)
- ✅ All dependencies are in `requirements.txt`
- ✅ `.env` file has your API keys
- ✅ `.gitignore` properly excludes secrets
- ✅ Git repository is initialized
- ✅ Streamlit config files exist

### Example output:
```
📂 Checking files...
  ✅ ui.py
  ✅ main.py
  ✅ requirements.txt
  ✅ app/ingestion.py
  ... more files ...

🔐 Checking secrets...
  ✅ .env file found (local)
  ✅ GROQ_API_KEY found in .env
  ✅ SUPABASE_URL found in .env
  ✅ SUPABASE_KEY found in .env

📦 Checking requirements.txt...
  ✅ streamlit
  ✅ pypdf
  ✅ sentence-transformers
  ... more packages ...

✅ Summary: 4/4 checks completed

🎉 Your project is ready for Streamlit Cloud deployment!
```

### If there are errors:
Fix them before continuing to Step 2.

---

## 🛠️ Option B: SETUP Your Project (Interactive Helper)

### What it does:
Automatically creates and configures files needed for deployment. It will:
- Create `.env.template` (safe to share)
- Update `.gitignore` with required entries
- Initialize Git repository
- Generate deployment checklist
- Show you how to get API keys

### How to run:
```bash
python setup_deployment.py
```

### What happens:
```
1️⃣  Creating configuration files...
  ✅ Created .env.template
  ✅ Added .env to .gitignore
  ✅ Added .streamlit/secrets.toml to .gitignore

2️⃣  Git setup...
  ✅ Initialized git repository

🔑 Get Your API Keys
  Groq API → https://console.groq.com
  Supabase → https://supabase.com
  GitHub → https://github.com

📝 GitHub Setup
  Instructions to set up GitHub remote...

3️⃣  Generating checklist...
  ✅ Saved to DEPLOYMENT_CHECKLIST.txt
```

---

## 📋 MANUAL Step 1 (If you want to do it manually):

If you prefer not to run Python scripts, do this manually:

### 1. Create `.env` file
In your project folder (same place as `ui.py`), create a file named `.env`:

**On Windows (PowerShell):**
```powershell
# Create empty .env file
New-Item -Path ".env" -ItemType File

# Open in Notepad to edit
notepad .env
```

**Then add your API keys:**
```
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_TIMEOUT_SECS=120
GROQ_MAX_RETRIES=5
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-actual-supabase-key
```

### 2. Check `.gitignore`
Make sure it has these lines:
```
.env
.streamlit/secrets.toml
__pycache__/
*.pyc
venv/
```

### 3. Check `requirements.txt`
Make sure it has all packages:
```
streamlit>=1.28.0
pypdf>=3.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.0
numpy>=1.24.0
groq>=0.4.0
python-dotenv>=1.0.0
supabase>=2.0.0
```

### 4. Check `.streamlit` folder
Make sure you have:
- `.streamlit/config.toml` ✅ (already created)
- `.streamlit/secrets.toml.template` ✅ (already created)

---

## ✅ How to Know Step 1 is Complete

You're ready for Step 2 when:

- ✅ You have a `.env` file with your API keys
- ✅ You have all files in the project folder
- ✅ `.gitignore` excludes `.env` and secrets
- ✅ `.streamlit/config.toml` exists
- ✅ `requirements.txt` has all dependencies

---

## 🎯 What's Next?

After Step 1 is complete, you're ready for **Step 2: Push to GitHub**

Go to → [QUICK_DEPLOY.md](QUICK_DEPLOY.md) Step 2
