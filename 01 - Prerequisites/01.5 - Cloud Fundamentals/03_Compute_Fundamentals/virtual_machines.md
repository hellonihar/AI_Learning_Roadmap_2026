# Virtual Machines (EC2 / GCE / Azure VMs)

## Key Concepts
- **Instance type**: CPU, memory, GPU, network performance
- **AMI / Image**: pre-configured OS template (Ubuntu, Deep Learning AMI)
- **Instance store**: temporary SSD attached to host (ephemeral)
- **EBS volume**: persistent block storage (survives stop/start)

## Sizing
| Type Family | Use Case | Examples |
|---|---|---|
| General purpose | Web servers, dev | t3, m5, n2-standard |
| Compute optimized | Batch processing | c5, c2-standard |
| Memory optimized | In-memory DB, caching | r5, m1-mega |
| GPU / Accelerator | ML training, inference | p4d, p5, g5, a3-high |

## For AI Engineers
- Use **Deep Learning AMI** (pre-installed CUDA, PyTorch, TensorFlow)
- Attach **EBS gp3** for boot volume, **FSx for Lustre** for high-throughput training data
- Use **spot instances** for training (up to 90% discount, can be interrupted)
- Never use `t` burstable instances for production ML workloads
