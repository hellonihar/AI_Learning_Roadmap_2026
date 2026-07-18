# Advanced Git

```bash
# Stash (save work in progress)
git stash                     # save dirty state
git stash pop                 # restore latest stash
git stash list                # list stashes
git stash apply stash@{2}     # apply specific stash

# Cherry-pick (apply a specific commit)
git cherry-pick <commit-hash>

# Revert vs Reset
git revert <commit-hash>       # safe: creates new commit undoing changes
git reset --soft <hash>        # undo commits, keep changes staged
git reset --hard <hash>        # DANGER: discard everything after hash

# Bisect (find the commit that broke something)
git bisect start
git bisect bad                 # current commit is bad
git bisect good <known-good>   # known good commit
# git checks out midpoint → test → mark good/bad → repeat

# Tags
git tag v1.0.0                # lightweight tag
git tag -a v1.0.0 -m "v1.0"  # annotated tag
git push --tags

# .gitignore for AI projects
gitignore_patterns = """
data/
models/
*.pkl
*.pt
*.h5
.env
__pycache__/
.ipynb_checkpoints/
.vscode/
"""
```
