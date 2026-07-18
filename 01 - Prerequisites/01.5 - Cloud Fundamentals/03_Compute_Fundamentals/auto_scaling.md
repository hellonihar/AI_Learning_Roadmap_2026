# Auto-scaling

Auto-scaling automatically adjusts the number of compute instances based on demand.

## How It Works
- Define a **launch template** (AMI, instance type, security groups)
- Set **scaling policies** (CPU > 70% → add 2 instances)
- Set **min / max / desired** capacity
- Health checks replace unhealthy instances automatically

## Scaling Policies
| Type | Example |
|---|---|
| Target tracking | Keep CPU at 50% |
| Step scaling | Add 2 instances when CPU > 70%, add 4 when CPU > 90% |
| Scheduled | Scale up 8 AM every weekday |

## For AI Engineers
- Batch inference jobs: scale to 100+ instances during job, scale to 0 after
- Real-time endpoints: keep minimum 2 for HA, scale based on latency or request count
- Training: auto-scaling less common (training is steady-state), but spot fleet can replace interrupted instances
