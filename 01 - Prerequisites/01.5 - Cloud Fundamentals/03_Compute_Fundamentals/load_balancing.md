# Load Balancing

Distributes incoming traffic across multiple targets (instances, containers, Lambda).

## Types
| Type | Layer | Best For |
|---|---|---|
| Application LB (ALB) | HTTP/HTTPS (L7) | Web apps, microservices, path-based routing |
| Network LB (NLB) | TCP/UDP (L4) | Low latency, static IP, gRPC |
| Gateway LB | L3 | Security appliances, traffic inspection |

## Key Features
- **Health checks**: automatically route away from unhealthy targets
- **Sticky sessions**: route same user to same target (if needed)
- **SSL termination**: offload TLS decryption
- **Cross-zone balancing**: evenly distribute across AZs

## For AI Engineers
- Place model inference endpoints behind an ALB
- Use path-based routing for multi-model endpoints (`/model-a`, `/model-b`)
- Health check endpoint should validate model is loaded and ready (`GET /health`)
