# Git Basics

```bash
# Configuration
git config --global user.name "Your Name"
git config --global user.email "you@email.com"

# Starting a repo
git init                    # new repo
git clone <url>             # copy existing repo

# Everyday workflow
git status                  # what's changed?
git diff                    # see unstaged changes
git add <file>              # stage a file
git add -A                  # stage all changes
git commit -m "message"     # snapshot staged changes
git log --oneline --graph   # history

# Syncing with remote
git push origin main        # upload commits
git pull                    # download + merge
git fetch                   # download without merge
```

## Typical AI Project Walkthrough

```bash
git clone https://github.com/your-org/ml-project
cd ml-project
# ... modify code ...
git add -A
git commit -m "Add data preprocessing pipeline"
git push
```
