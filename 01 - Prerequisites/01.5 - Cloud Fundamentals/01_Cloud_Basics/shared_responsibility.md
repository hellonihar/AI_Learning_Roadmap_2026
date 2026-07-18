# Shared Responsibility Model

```
Responsibility                    | Provider | Customer
──────────────────────────────────┼──────────┼──────────
Physical security / hardware      |    ✅    |
Network infrastructure            |    ✅    |
Hypervisor / virtualization       |    ✅    |
Managed service runtime           |    ✅    |
──────────────────────────────────┼──────────┼──────────
OS patching (VM)                  |          |    ✅
Application code & config         |          |    ✅
Data encryption / access          |          |    ✅
Network traffic controls          |          |    ✅
IAM policies & secrets            |          |    ✅
```

## Key Rules
- Provider is responsible **of** the cloud (hardware, networking, data centers)
- Customer is responsible **in** the cloud (data, access, configurations)
- For managed services (PaaS), provider handles more
- For IaaS (VMs), customer handles OS and above
- **Biggest risk**: misconfigured permissions exposing data
