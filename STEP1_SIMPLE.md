# 🚀 EASIEST WAY - Step by Step

## Step 1: Prepare ⚙️

### **Choice A: Let the Script Check Everything (EASIEST)**

Open PowerShell in your project folder and run:

```powerShell
python verify_deployment.py
```

**Expected result:**
```
✅ Summary: 4/4 checks completed
🎉 Your project is ready for Streamlit Cloud deployment!
```

If you see ✅ everywhere, you're done with Step 1!

---

### **Choice B: Let the Script Setup Everything (AUTOMATED)**

```powershell
python setup_deployment.py
```

This will:
- ✅ Create `.env.template`
- ✅ Setup `.gitignore`
- ✅ Initialize Git
- ✅ Show you where to get API keys
- ✅ Create deployment checklist

---

### **Choice C: Manual Setup (5 minutes)**

If scripts don't work for you, do this manually:

#### 1️⃣ Get Your API Keys (5 min)

Visit these websites and create free accounts:

**Groq API:**
- Go to: https://console.groq.com
- Sign up (free)
- Go to API Keys
- Copy your key (looks like: `gsk_xxxxxxxxxxxx`)

**Supabase:**
- Go to: https://supabase.com
- Sign up (free)
- Create new project
- Go to Settings → API
- Copy URL and anon key

**GitHub:**
- Go to: https://github.com
- Sign up (free)

#### 2️⃣ Create `.env` file

In PowerShell, navigate to your project:

```powershell
cd c:\Users\prash\Desktop\rag-system
```

Create `.env` file:

```powershell
notepad .env
```

Paste this (with YOUR keys):

```
GROQ_API_KEY=gsk_your_key_from_groq
SUPABASE_URL=https://your-url.supabase.co
SUPABASE_KEY=your-key-from-supabase
```

Save and close.

---

## ✅ You're Done with Step 1 When:

- [ ] `.env` file exists in your project folder
- [ ] You see the file when you run: `ls` or `dir`
- [ ] All API keys are filled in `.env`

---

## 🎯 Next: Step 2

Once Step 1 is complete, go to **Step 2: Push to GitHub**

```bash
cd c:\Users\prash\Desktop\rag-system
git add .
git commit -m "Ready for deployment"
git push origin main
```

Then continue to **Step 3: Deploy on Streamlit Cloud**

---

## 🆘 Troubleshooting Step 1

**Q: I can't find where to get API keys**
- Groq: https://console.groq.com/keys
- Supabase: https://app.supabase.com → Your Project → Settings → API

**Q: The script says files are missing**
- Check you're in the right folder: `pwd` or `Get-Location`
- Make sure you're in: `c:\Users\prash\Desktop\rag-system`

**Q: `.env` file won't save**
- Use: `echo "KEY=VALUE" > .env` in PowerShell
- Or open Terminal and use: `cat > .env` then paste content

**Q: Where does `.env` go?**
- Same folder as `ui.py`
- Example: `c:\Users\prash\Desktop\rag-system\.env`

---

**Ready for Step 2?** → Go to [QUICK_DEPLOY.md](QUICK_DEPLOY.md)
