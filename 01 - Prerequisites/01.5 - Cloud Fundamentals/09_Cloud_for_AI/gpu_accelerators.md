# GPU & Accelerator Instances

## Current GPU Instance Types (2026)
| Provider | Instance | GPU | VRAM | Use Case |
|---|---|---|---|---|
| AWS | p5.48xlarge | 8× H100 | 80GB each | Large model training |
| AWS | p4d.24xlarge | 8× A100 | 40GB each | Training, fine-tuning |
| AWS | g5.xlarge | 1× A10G | 24GB | Inference, fine-tuning |
| GCP | a3-high-gpu-8g | 8× H100 | 80GB each | Large model training |
| GCP | a2-highgpu-8g | 8× A100 | 40GB each | Training |
| GCP | g2-standard-4 | 1× L4 | 24GB | Inference, fine-tuning |
| Azure | ND H100 v5 | 8× H100 | 80GB each | Training |
| Azure | NCas T4 v3 | 1× T4 | 16GB | Inference, light training |

## Choosing an Instance
- **Training**: H100 > A100 > L40S — prioritize memory bandwidth and inter-GPU speed
- **Fine-tuning**: A10G / L4 / T4 are cost-effective
- **Inference**: depends on model size — 7B fits on T4 (FP16), 70B needs A100/H100

## Availability
- GPU instances often have quota limits — request increases early
- Some regions have no GPU capacity — check before designing architecture
- Spot GPU instances can save 60-90% but may be interrupted
- Reserve instances for guaranteed capacity on long-term projects
