# Collaboration Workflow

```bash
# Fork & clone
# 1. Fork on GitHub
# 2. Clone your fork
git clone https://github.com/your-username/project.git
git remote add upstream https://github.com/original/project.git

# Keep fork up to date
git fetch upstream
git checkout main
git merge upstream/main
git push

# Pull Request workflow
git checkout -b my-feature
# ... make changes ...
git add -A && git commit -m "Add feature"
git push origin my-feature
# Open PR on GitHub → code review → merge

# After PR is merged
git checkout main
git pull upstream main
git push origin main
```

## Code Review Tips for AI

- Review logic correctness, not just style
- Check for hardcoded paths, seeds, or configs
- Verify that experiment results are reproducible
- Ensure sensitive data (API keys, data files) isn't committed
