# Branching & Merging

```bash
# Branches
git branch                  # list branches
git branch feature-x        # create branch
git checkout feature-x      # switch to branch
git checkout -b feature-x   # create + switch

# Merging
git checkout main
git merge feature-x         # merge feature into main

# Resolving conflicts
# After merge conflict, edit files to resolve, then:
git add <resolved-files>
git commit -m "Resolve merge conflict"

# Rebase (cleaner history)
git checkout feature-x
git rebase main             # replay feature commits on top of main
```

## Branch Strategy for AI Projects

```
main          ────●────────────●────────────
                  \            /
feature/data   ────●──●──●────
feature/model  ────●──●──●──●────
```

- `main` — production-ready code
- `feature/*` — one branch per experiment or dataset change
- Never commit large models or data to git (use DVC or Git LFS)
