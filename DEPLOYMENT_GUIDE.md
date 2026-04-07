# 🚀 Deploy RAG System to Streamlit Cloud

## Step-by-Step Deployment Guide

### **Step 1: Prepare Your GitHub Repository**

#### 1a. Create a GitHub Account (if you don't have one)
- Visit https://github.com/signup
- Create account and verify email

#### 1b. Create a New GitHub Repository
- Go to https://github.com/new
- Repository name: `rag-system` (or your choice)
- Set to **Public** (free tier requirement)
- Click "Create repository"

#### 1c. Push Your Code to GitHub
```bash
cd c:\Users\prash\Desktop\rag-system

# Initialize git repo (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial RAG System commit"

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/rag-system.git

# Push to main branch
git branch -M main
git push -u origin main
```

**⚠️ Important:** Make sure `.gitignore` includes:
```
.env
.streamlit/secrets.toml
data/
indexes/
__pycache__/
```

---

### **Step 2: Deploy on Streamlit Cloud**

#### 2a. Sign Up for Streamlit Cloud
- Visit https://share.streamlit.io/
- Click "Sign up with GitHub"
- Authorize Streamlit to access your GitHub account
- Sign in with your GitHub credentials

#### 2b. Create a New App
- Click **"New app"** button (top right)
- Select:
  - **Repository**: YOUR_USERNAME/rag-system
  - **Branch**: main
  - **Main file path**: `ui.py`
- Click **Deploy**

**Expected Deployment Time:** 3-5 minutes

---

### **Step 3: Add Secrets in Streamlit Cloud**

After deployment, your app will show errors because secrets are missing.

#### 3a. Access App Settings
- Open your deployed app on Streamlit Cloud
- Click **☰ menu** (top right)
- Select **Settings**
- Go to **Secrets** tab

#### 3b. Add Your Secrets
Copy-paste these (with YOUR actual values):

```toml
# Groq API - Get from https://console.groq.com
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
GROQ_TIMEOUT_SECS = "120"
GROQ_MAX_RETRIES = "5"

# Supabase - Get from https://supabase.com
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Cloud Storage (optional)
CLOUD_STORAGE_BUCKET = "your-bucket-name"
```

#### 3c. Click "Save"

Your app will automatically restart with the secrets loaded ✅

---

### **Step 4: Upload PDF Files**

#### 4a. Option 1: Upload via Web UI
- Your Streamlit app has a file upload feature
- Users can upload PDFs directly through the interface

#### 4b. Option 2: Pre-load PDFs in the `data/` folder

**⚠️ Issue:** Your `.gitignore` excludes `data/` and `*.pdf`, so they won't be pushed to GitHub.

**Solution A:** Keep `.gitignore` but upload PDFs via Streamlit UI
- Users upload files when they access the app
- PDFs are processed and indexed on-demand

**Solution B:** Remove data exclusion for cloud deployment
Create `.gitignore-cloud`:
```bash
# Exclude sensitive files only
.env
.streamlit/secrets.toml
__pycache__/
indexes/
```

Then:
```bash
# Rename for cloud
mv .gitignore .gitignore-local
mv .gitignore-cloud .gitignore
git add .gitignore
git commit -m "Enable PDF files for cloud"
git push
```

---

### **Step 5: Monitor Your Deployment**

#### 5a. View Logs
- App page → **☰ menu** → **Get logs**
- Check for errors in:
  - Model downloads
  - API connections
  - Database connections

#### 5b. Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "GROQ_API_KEY not found" | Secrets not added | Add in Settings → Secrets |
| "Connection timeout" | Groq API down | Check https://status.groq.com |
| "Supabase connection failed" | Wrong credentials | Verify SUPABASE_URL + KEY |
| "Model download failed" | Slow internet | Streamlit retries automatically |
| "Out of memory" | Large PDF batch | Reduce `batch_size` in `embedding.py` |

---

### **Step 6: Custom Domain (Optional Paid Feature)**

1. Subscribe to "Streamlit+ Team" ($99/month)
2. Go to **Workspace settings** → **Custom domain**
3. Add your domain (e.g., `rag.yourcompany.com`)
4. Update DNS records
5. Enable SSL

---

## 🔧 Configuration for Cloud Deployment

### **Memory & Resource Limits**
- Streamlit Cloud: 1 GB RAM (shared tier)
- FAISS + models: ~800 MB
- Available for temp data: ~200 MB

**Optimization Tips:**
```python
# Reduce batch size for embeddings
batch_size = 32  # Instead of 64

# Use smaller embedding model
model_name = "BAAI/bge-small-en"  # Already using this ✓

# Clear cache between sessions
st.cache_resource.clear() if st.button("Clear Cache") else None
```

### **Timeout Handling**
Streamlit Cloud has a **timeout of 3 hours per session**. For long-running operations:

```python
# In ui.py, add:
if st.session_state.get("still_processing"):
    st.warning("⏱️ Processing large file... please wait")
    st.info("If this times out, the file may be too large. Try splitting it.")
```

---

## 📊 Performance Expectations on Streamlit Cloud

| Operation | Time |
|-----------|------|
| App startup | 15-30s |
| Load indexes | 5s |
| Query embedding | 2s |
| Vector search | 0.1s |
| BM25 search | 0.05s |
| Reranking | 0.5s |
| LLM response | 1-3s |
| **Total per query** | **~10-20s** |

---

## 🔒 Security Best Practices

### ✅ Do's
- ✅ Store secrets ONLY in Streamlit Cloud dashboard
- ✅ Use `.env.template` to show required keys
- ✅ Rotate API keys quarterly
- ✅ Use Supabase Row-Level Security (RLS)
- ✅ Enable Supabase Auth if adding user login

### ❌ Don'ts
- ❌ Commit `.env` or `secrets.toml` to GitHub
- ❌ Hardcode API keys in `ui.py`
- ❌ Use public database keys (use anon key only)
- ❌ disable `enableXsrfProtection` in production

---

## 📈 Monitoring & Analytics

### **View App Activity**
- Streamlit Cloud dashboard → **Analytics** tab
- See:
  - Daily active users
  - Session duration
  - Error rates
  - Deployment status

### **Set Up Alerts**
- Go to **Settings** → **Notifications**
- Enable email for:
  - App crashes
  - Resource limits exceeded
  - Deployment failures

---

## 🚀 Advanced: Custom Domain + CI/CD

### **Auto-Deploy on GitHub Push**
Streamlit automatically redeploys when you push to `main`:

```bash
# Make changes locally
git add .
git commit -m "Feature: Query rewriting"
git push origin main

# Streamlit Cloud automatically redeploys (3-5 min)
```

### **Multiple Environments**
Create branches for staging:
```bash
git checkout -b staging
# ... make changes
git push origin staging

# Deploy separately in Streamlit Cloud
# Repository → staging branch
```

---

## 💾 Data Persistence

### **Session Storage (Supabase)**
- ✅ Chat history saved automatically
- ✅ Survives app restarts
- ✅ Accessible across devices

### **Indexes (FAISS + BM25)**
- 🔄 Persisted in `indexes/` folder (local)
- ⚠️ **Problem:** Local storage is ephemeral on Streamlit Cloud
- ✅ **Solution:** Download from cloud on startup (`app/cloud_storage.py`)

### **Recommended Architecture**
```
GitHub:        Code + main.py + ui.py (no data)
      ↓
Streamlit Cloud: Runs app + creates indexes in memory
      ↓
Supabase:      Store sessions + conversations
      ↓
S3/Cloud Storage: Backup FAISS/BM25 indexes
```

---

## 🐛 Troubleshooting Deployment

### **App won't start**
```bash
# Check stdout logs
tail -f ~/.streamlit/logs/streamlit.log

# Common causes:
# 1. Missing required environment variable
# 2. Importing module not in requirements.txt
# 3. Incompatible package version
```

### **Slow startup (>1 min)**
- Models are downloading (first run only)
- Supabase connection slow
- Too many imports at module level

### **Out of memory errors**
```python
# Reduce in app/embedding.py
batch_size = 16  # From 64

# Reduce in app/pipeline.py
top_k_retrieve = 10  # From 20
```

### **PDF uploads fail**
Check `maxUploadSize` in `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200  # MB
```

---

## 📞 Support & Resources

| Resource | Link |
|----------|------|
| Streamlit Cloud Docs | https://docs.streamlit.io/streamlit-cloud |
| GitHub | https://github.com/pushpreof/rag-system |
| Groq Console | https://console.groq.com |
| Supabase Dashboard | https://app.supabase.com |
| Streamlit Forums | https://discuss.streamlit.io |

---

## ✅ Deployment Checklist

Before deploying:
- [ ] GitHub account created
- [ ] Code pushed to GitHub (main branch)
- [ ] `.gitignore` updated with `.streamlit/secrets.toml`
- [ ] `requirements.txt` has all dependencies with versions
- [ ] `.streamlit/config.toml` exists
- [ ] `ui.py` is the entry point
- [ ] Groq API key obtained
- [ ] Supabase account + keys ready

After deployment:
- [ ] Added secrets in Streamlit Cloud dashboard
- [ ] App loads without errors (check logs)
- [ ] Can query and get LLM responses
- [ ] Chat history saves to Supabase
- [ ] PDF upload works (if needed)

---

## 🎉 You're Deployed!

Your app is now live at:
```
https://YOUR_USERNAME-rag-system.streamlit.app/
```

Share this link with others to use your RAG system! 🚀

---

**Last Updated:** April 2026  
**Streamlit Cloud Version:** 1.28+  
**Status:** Production-Ready
