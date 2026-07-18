# Code Review Assistants

AI tools that automate code review on pull requests.

## CodeRabbit

Automatically reviews every PR with inline comments.

| Feature | Detail |
|---|---|
| Inline comments | Comments on specific lines in the PR |
| Summaries | Generates PR summary description |
| Security scanning | Detects vulnerable patterns |
| Performance suggestions | Identifies slow code paths |
| Integration | GitHub, GitLab, Bitbucket |

**Prompt example (in PR):** "Does this PR handle the case where the user has no prior history?"

## Copilot Code Review

GitHub's built-in AI reviewer for pull requests.

**Strengths**: Free, integrated into GitHub, works on any public repo.

## CodeBall

Specialized for security and compliance in code reviews.

## What AI Reviewers Catch

| Category | Example |
|---|---|
| Security | Hardcoded secrets, SQL injection, path traversal |
| Bugs | Off-by-one, missing error handling, type mismatches |
| Performance | N+1 queries, missing indexes, O(n²) where O(n) works |
| Style | Inconsistent naming, missing docstrings, dead code |
| Logic | Wrong operator, incorrect boundary check, missing edge case |

## Limitations

- Cannot test the code (no execution)
- May miss business logic errors
- Can have false positives
- Won't catch issues specific to your domain
