# Identifying Security Issues

AI models can generate insecure code. Always review for these patterns.

## Critical Red Flags

### 1. Code Injection
```python
# ❌ NEVER DO THIS
result = eval(user_input)                    # code injection
os.system(f"rm -rf {user_path}")            # command injection
subprocess.run(cmd, shell=True)             # shell injection
exec(user_code)                              # full code execution
```

### 2. Path Traversal
```python
# ❌ Vulnerable
open(f"/data/{user_filename}").read()       # ../../etc/passwd

# ✅ Safe
import os
base = Path("/data")
path = (base / user_filename).resolve()
if not str(path).startswith(str(base)):
    raise ValueError("Invalid path")
```

### 3. Hardcoded Secrets
```python
# ❌ NEVER
API_KEY = "sk-abc123..."
password = "admin123"
DB_URL = "postgres://user:pass@host/db"

# ✅ Use environment variables
import os
API_KEY = os.environ["API_KEY"]
```

### 4. SQL Injection
```python
# ❌ Vulnerable
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# ✅ Safe
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

## Security Prompt

After generating code, ask AI:
```
"Review this code for security vulnerabilities:
- Injection attacks (SQL, command, code)
- Hardcoded credentials
- Path traversal
- Insecure deserialization
- Missing input validation
```
