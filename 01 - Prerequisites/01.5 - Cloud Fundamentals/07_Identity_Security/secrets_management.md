# Secrets Management

Never hardcode secrets in code, config files, or environment variables in plain text.

## Cloud Secrets Services
| Provider | Service | Use Case |
|---|---|---|
| AWS | Secrets Manager | Auto-rotation, cross-account access |
| AWS | Parameter Store (SSM) | Free tier, hierarchy, plain or secret |
| GCP | Secret Manager | Managed, versioned, audit logging |
| Azure | Key Vault | Certificates, secrets, encryption keys |

## Best Practices
```python
# BAD
api_key = "sk-abc123..."  # never

# GOOD
import os
api_key = os.environ["OPENAI_API_KEY"]  # from environment

# BETTER
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name: str) -> str:
    client = boto3.client("secretsmanager")
    try:
        resp = client.get_secret_value(SecretId=secret_name)
        return resp["SecretString"]
    except ClientError as e:
        raise e

# BEST (for dev)
# .env file with python-dotenv, .env in .gitignore
```

## Secret Rotation
- Use managed rotation for DB credentials (RDS + Secrets Manager)
- Rotate API keys regularly (every 90 days)
- Use short-lived tokens (STS, OIDC) instead of long-lived keys when possible
