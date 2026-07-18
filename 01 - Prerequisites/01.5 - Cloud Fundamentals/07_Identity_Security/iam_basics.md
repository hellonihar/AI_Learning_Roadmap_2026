# Identity & Access Management (IAM)

IAM lets you control who can do what with your cloud resources.

## Core Components
- **User**: individual person or service account
- **Group**: collection of users (easier management)
- **Role**: set of permissions that can be assumed (no credentials)
- **Policy**: JSON document defining permissions (what, which resource, when)
- **Service Account** (GCP): identity for apps/services

## Policy Structure (AWS example)
```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["s3:GetObject", "s3:PutObject"],
        "Resource": "arn:aws:s3:::my-bucket/*",
        "Condition": {
            "IpAddress": {"aws:SourceIp": "10.0.0.0/16"}
        }
    }]
}
```

## Best Practices
- **Least privilege**: grant minimum permissions needed
- **Use roles, not users**: apps assume roles, they don't use user credentials
- **Use groups**: assign policies to groups, add users to groups
- **Rotate keys**: regularly rotate access keys
- **Enable MFA**: for all human users with console access
