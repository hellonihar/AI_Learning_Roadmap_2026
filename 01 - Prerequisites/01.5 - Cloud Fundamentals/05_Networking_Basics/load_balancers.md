# Load Balancers (Detailed)

## ALB vs NLB
| Feature | ALB | NLB |
|---|---|---|
| Layer | L7 (HTTP/HTTPS) | L4 (TCP/UDP) |
| Content-based routing | ✅ (path, host, headers) | ❌ |
| WebSocket / gRPC | ✅ | ✅ |
| Static IP | ❌ (uses LB hostname) | ✅ (per AZ) |
| Latency | 2-3ms | <1ms |
| TLS termination | ✅ | ✅ |

## ALB Routing Examples
```yaml
# Path-based routing
/model-a/*    → target group A (gpt-2 instance)
/model-b/*    → target group B (gpt-4 instance)
/health       → target group C (health service)

# Host-based routing
api.example.com  → target group API
admin.example.com → target group Admin
```

## For AI Engineers
- Use ALB for REST/HTTP model endpoints
- Use NLB for gRPC inference with low latency requirements
- Configure health check path as `/health` returning 200 if model is loaded
- Use stickiness only if needed (most inference is stateless)
