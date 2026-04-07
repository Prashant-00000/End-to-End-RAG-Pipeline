# 🚀 Streamlit Cloud Deployment - Quick Start

## **5-Minute Quick Deploy**

### 1️⃣ Push to GitHub
```bash
cd c:\Users\prash\Desktop\rag-system
git add .
git commit -m "Deploy to Cloud"
git push origin main
```

### 2️⃣ Deploy on Streamlit Cloud
- Go: https://share.streamlit.io/
- **Sign in** with GitHub
- **New app**
- Select: `YOUR_USERNAME/rag-system`
- Main file: `ui.py`
- Click **Deploy**

### 3️⃣ Add Secrets (Wait 3 mins for app to load first)
- App menu (☰) → **Settings** → **Secrets**
- Paste:
```toml
GROQ_API_KEY = "gsk_..."
SUPABASE_URL = "https://....supabase.co"
SUPABASE_KEY = "eyJ..."
```
- Click **Save**

### ✅ Done! 
Your app is at: `https://YOUR_USERNAME-rag-system.streamlit.app/`

---

## 🔑 Required API Keys

| Service | Where to Get | Free Plan? |
|---------|-------------|-----------|
| **Groq** | https://console.groq.com | ✅ Yes |
| **Supabase** | https://supabase.com | ✅ Yes (500GB) |
| **GitHub** | https://github.com | ✅ Yes |

---

## ⚠️ Common Issues & Fixes

```
Error: "GROQ_API_KEY not found"
→ Add in Settings → Secrets

Error: "Supabase connection failed"  
→ Check SUPABASE_URL and SUPABASE_KEY in Secrets

Error: "Module not found"
→ Add to requirements.txt, push to GitHub

Error: "Out of memory"
→ Use smaller batch size or fewer PDFs
```

---

## 📦 Pre-Deployment Checklist

- [ ] GitHub repo created and code pushed
- [ ] `.env` NOT in GitHub (in .gitignore)
- [ ] Groq API key ready
- [ ] Supabase project created
- [ ] `requirements.txt` has all packages

---

## 🌐 Your Live App URL

```
https://YOUR_USERNAME-rag-system.streamlit.app/
```

Share this link with anyone to use your RAG system!

---

## 📞 Get Help

- Streamlit Cloud Issues: https://discuss.streamlit.io
- Groq API Issues: https://console.groq.com/help
- Supabase Issues: https://supabase.com/support
