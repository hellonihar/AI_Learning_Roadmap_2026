# Terminal Basics

```bash
# Navigation
pwd              # print working directory
ls -la           # list files
cd <path>        # change directory
mkdir -p a/b/c   # create nested directories
rm -rf <dir>     # remove directory

# Environment variables
export MY_KEY="abc123"
echo $MY_KEY
echo "Path: $PATH"

# PATH
export PATH="$HOME/.local/bin:$PATH"

# Aliases (add to ~/.bashrc or ~/.zshrc)
alias ll="ls -la"
alias gs="git status"
alias gc="git commit -m"
alias gp="git push"

# Process management
ps aux                    # list processes
kill -9 <PID>             # force kill
nohup python train.py &   # run in background
screen -S train           # persistent session
tmux new -s train         # alternative to screen

# .bashrc / .zshrc essentials
export EDITOR="code --wait"
export PIP_REQUIRE_VIRTUALENV=true
```
