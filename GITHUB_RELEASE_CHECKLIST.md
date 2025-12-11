# GitHub Release Checklist
**XAU Trading Bot - Pre-Release Security & Quality Checklist**

Use this checklist to ensure your repository is secure and ready for public release.

---

## 🔒 SECURITY (CRITICAL - Complete Before git push)

### Step 1: Review Security Audit
- [ ] Read `SECURITY_AUDIT_REPORT.md` completely
- [ ] Understand all identified security risks
- [ ] Review sensitive files list

### Step 2: Run Cleanup Script
```bash
# Dry run first to see what will be deleted
python cleanup_before_git.py --dry-run

# If everything looks good, run actual cleanup
python cleanup_before_git.py
```

**What gets deleted:**
- [ ] All `.log` files (contains MT5 account number 67138076)
- [ ] Large `.csv` data files (optional, can regenerate)
- [ ] `output.txt` files
- [ ] Python cache (`__pycache__`)

### Step 3: Verify .gitignore
- [ ] `.gitignore` file exists
- [ ] Contains `*.log` pattern
- [ ] Contains `*.csv` pattern (optional)
- [ ] Contains Python artifacts (`__pycache__`, `*.pyc`)
- [ ] Test: Run `git status` and ensure no `.log` files listed

### Step 4: Code Review for Hardcoded Secrets
- [ ] No API keys in source code
- [ ] No passwords in source code
- [ ] No email addresses in source code
- [ ] No MT5 credentials hardcoded
- [ ] No IP addresses or server URLs

### Step 5: Sanitize Logging (Optional but Recommended)
Consider updating these files to redact account numbers in logs:
- [ ] `bot.py` lines 156, 158 - Account number logging
- [ ] `newmut05/bot.py` lines 117, 120
- [ ] `mutation_1/bot_new.py` lines 71, 73

**Suggested change:**
```python
# Before:
logging.info(f"Trading is allowed for account {account_info.login}")

# After:
account_redacted = f"***{str(account_info.login)[-4:]}"
logging.info(f"Trading is allowed for account {account_redacted}")
```

---

## 📝 DOCUMENTATION

### README
- [ ] `README.md` exists and is complete
- [ ] Security warnings are prominent
- [ ] Installation instructions are clear
- [ ] Usage examples are provided
- [ ] Architecture diagram is included
- [ ] Contact/support information added

### License
- [ ] `LICENSE` file exists
- [ ] License type is appropriate (MIT recommended)
- [ ] Trading disclaimer is included

### Additional Documentation
- [ ] `SECURITY_AUDIT_REPORT.md` reviewed (can keep or exclude)
- [ ] Consider adding `CONTRIBUTING.md` for contribution guidelines
- [ ] Consider adding `CHANGELOG.md` for version history

---

## 🧪 CODE QUALITY

### Testing
- [ ] All models train successfully
- [ ] Backtesting runs without errors
- [ ] Pipeline script (`run_pipeline.py`) executes fully
- [ ] No syntax errors in any `.py` files

### Code Organization
- [ ] Remove commented-out code
- [ ] Remove debug print statements
- [ ] Ensure consistent code style
- [ ] Add docstrings to main functions

### Dependencies
- [ ] Create `requirements.txt` with all dependencies:
```bash
pip freeze > requirements.txt
```
- [ ] Test installation in fresh virtual environment

---

## 🐙 GIT REPOSITORY SETUP

### Step 1: Initialize Repository
```bash
cd C:\Users\X\Documents\Pythonbotforex\xaubot_v4.8.8
git init
```

### Step 2: Review Files Before Adding
```bash
# See what will be added
git status

# Check .gitignore is working
git status | grep -E "(\.log|\.csv)"  # Should be empty
```

### Step 3: Stage Files
```bash
git add .
```

### Step 4: Review Staged Files (CRITICAL)
```bash
# List all files that will be committed
git ls-files

# Check for sensitive files
git ls-files | grep -E "(\.log|67138076|password|secret|api_key)"

# If any sensitive files are found, unstage them:
# git reset HEAD <filename>
# git rm --cached <filename>
```

### Step 5: Initial Commit
```bash
git commit -m "Initial commit: XAU Trading Bot v4.8.8

- Bidirectional LSTM + Transformer models for forex prediction
- Multi-timeframe analysis (M1-D1)
- MetaTrader 5 integration
- Automated pipeline with backtesting
- Security-first configuration
"
```

### Step 6: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `xau-trading-bot` (or your choice)
3. Description: "AI-powered automated forex trading bot for XAU/USD"
4. **Public** or Private (your choice)
5. **DO NOT** initialize with README (you already have one)
6. Create repository

### Step 7: Connect Local to Remote
```bash
# Replace with your actual repository URL
git remote add origin https://github.com/yourusername/xau-trading-bot.git

# Verify remote
git remote -v
```

### Step 8: Push to GitHub
```bash
# Push to main branch
git branch -M main
git push -u origin main
```

---

## ✅ POST-RELEASE VERIFICATION

### GitHub Web Interface Checks
- [ ] Repository loads correctly
- [ ] README.md displays properly
- [ ] No sensitive files visible in file browser
- [ ] Check commit history for accidentally committed secrets
- [ ] License file is recognized by GitHub

### Search for Leaks
Search repository on GitHub for:
- [ ] Your account number: `67138076`
- [ ] Keywords: `password`, `secret`, `api_key`, `token`
- [ ] Email addresses
- [ ] IP addresses

**If found:** Immediately remove repository and clean git history!

### Repository Settings
- [ ] Add description and tags
- [ ] Add topics: `forex`, `trading-bot`, `machine-learning`, `tensorflow`, `metatrader5`
- [ ] Enable/disable Issues (recommended: enable)
- [ ] Enable/disable Discussions (optional)
- [ ] Consider adding GitHub Actions for CI/CD

### Security Settings (GitHub)
- [ ] Enable Dependabot alerts
- [ ] Enable secret scanning (if available)
- [ ] Enable code scanning (CodeQL)
- [ ] Review security policy

---

## 🎯 OPTIONAL ENHANCEMENTS

### GitHub Features
- [ ] Add repository badges to README (build status, license, etc.)
- [ ] Create GitHub Actions workflow for testing
- [ ] Add issue templates
- [ ] Add pull request template
- [ ] Create SECURITY.md with vulnerability reporting process

### Git LFS for Large Files (Optional)
If you want to keep model files in repository:
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
git push
```

### Pre-commit Hooks (Prevent Future Leaks)
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Check for log files
if git diff --cached --name-only | grep -E "\.log$"; then
    echo "ERROR: Attempting to commit .log files!"
    echo "Run: git reset HEAD *.log"
    exit 1
fi

# Check for account numbers
if git diff --cached | grep -E "67138076"; then
    echo "ERROR: Attempting to commit account number!"
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
```
Make executable: `chmod +x .git/hooks/pre-commit`

### Documentation
- [ ] Add CHANGELOG.md
- [ ] Add CONTRIBUTING.md
- [ ] Add code examples
- [ ] Add troubleshooting guide
- [ ] Add FAQ section

---

## ⚠️ EMERGENCY: If Secrets Were Pushed

If you accidentally pushed sensitive data:

1. **IMMEDIATELY make repository private**
2. **DO NOT just delete files and commit** (history still contains secrets)
3. **Use BFG Repo-Cleaner or git-filter-branch:**
```bash
# Install BFG
# Download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove log files from history
java -jar bfg.jar --delete-files "*.log" xaubot_v4.8.8

# Remove account number from all history
java -jar bfg.jar --replace-text <(echo "67138076==>REDACTED") xaubot_v4.8.8

# Clean up
cd xaubot_v4.8.8
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (DANGEROUS)
git push --force
```
4. **Change any exposed credentials immediately**
5. **Consider creating a fresh repository with clean history**

---

## 📋 FINAL CHECKLIST

Before making repository public:
- [ ] All security checks passed
- [ ] No sensitive data in repository
- [ ] `.gitignore` is comprehensive
- [ ] README is complete and clear
- [ ] License file included
- [ ] Code is clean and documented
- [ ] Repository pushed successfully
- [ ] Post-release verification completed
- [ ] Security settings configured

---

## 🎉 READY FOR RELEASE!

Once all items are checked:
1. Make repository public (if currently private)
2. Share repository link
3. Monitor issues and pull requests
4. Update documentation as needed

**Congratulations on your public release! 🚀**

---

## 📞 Need Help?

- Review `SECURITY_AUDIT_REPORT.md` for detailed security analysis
- Check GitHub Docs: https://docs.github.com
- Git documentation: https://git-scm.com/doc
- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/

**Remember:** Security first, always!
