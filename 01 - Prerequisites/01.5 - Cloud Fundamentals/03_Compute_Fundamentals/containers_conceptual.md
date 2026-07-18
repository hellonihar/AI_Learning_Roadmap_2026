# Containers (Conceptual)

Containers package code + dependencies into a portable unit that runs the same everywhere.

## Docker Basics
- **Image**: read-only template (OS + app + deps)
- **Container**: running instance of an image
- **Dockerfile**: recipe to build an image
- **docker-compose.yml**: multi-container setup (app + DB + cache)

## Containers vs VMs
```
            VMs                    Containers
┌─────────────────────┐   ┌─────────────────────┐
│  App A │ App B      │   │  App A │ App B      │
│  Guest OS │ Guest OS│   │  Libs  │ Libs       │
│  Hypervisor         │   │  Container Runtime   │
│  Host OS            │   │  Host OS             │
└─────────────────────┘   └─────────────────────┘
- Each VM has full OS     - Share host OS kernel
- Heavy (GBs per VM)      - Lightweight (MBs per container)
- Slow to start           - Fast to start (seconds)
```

## For AI Engineers
- Containerize models for reproducible inference deployments
- Package training environment to avoid dependency conflicts
- Use container registries to version and share images
