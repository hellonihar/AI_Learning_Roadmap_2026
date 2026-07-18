# Regions & Availability Zones

## What is a Region?

A region is a geographic area containing 2 or more availability zones. Each cloud provider operates regions worldwide.

### Provider Region Examples

| Provider | Region Name | Location | Zones |
|---|---|---|---|
| **AWS** | `us-east-1` | N. Virginia | 6 |
| **AWS** | `eu-west-1` | Ireland | 3 |
| **AWS** | `ap-south-1` | Mumbai | 3 |
| **GCP** | `us-central1` | Iowa | 4 |
| **GCP** | `europe-west4` | Netherlands | 3 |
| **GCP** | `asia-southeast1` | Singapore | 3 |
| **Azure** | `eastus` | Virginia | 3 |
| **Azure** | `westeurope` | Netherlands | 3 |
| **Azure** | `southeastasia` | Singapore | 3 |

## How to Choose a Region

### 1. Latency (Proximity to Users)

Every 100ms of latency reduces user engagement. Deploy inference endpoints close to your users.

```
User in Mumbai → ap-south-1 (AWS)      ~5ms
User in Mumbai → us-east-1 (AWS)      ~200ms
User in Mumbai → eu-west-1 (AWS)      ~150ms
```

**Example**: If your users are in India, deploy inference in `ap-south-1` (Mumbai) or `ap-southeast-1` (Singapore), not `us-east-1`.

### 2. Compliance & Data Residency

Many countries require data to stay within borders.

| Requirement | Region Choice |
|---|---|
| EU user data (GDPR) | `eu-west-1` (Ireland), `eu-central-1` (Frankfurt), `europe-west4` (Netherlands) |
| India financial data | `ap-south-1` (Mumbai) |
| US healthcare (HIPAA) | `us-east-1` (N. Virginia), `us-west-2` (Oregon) |
| China operations | `cn-north-1` (Beijing) — requires separate account |
| Australia government | `ap-southeast-2` (Sydney) |

**Example**: A startup processing EU customer data with GDPR must use a European region. Training GPUs in `eu-central-1` (Frankfurt) and storing data in `eu-west-1` (Ireland) keeps everything within EU jurisdiction.

### 3. Service Availability

Not every service or instance type exists in every region.

| Service | us-east-1 | eu-west-1 | ap-south-1 |
|---|---|---|---|
| p5.48xlarge (8× H100) | ✅ | ✅ | ❌ |
| p4d.24xlarge (8× A100) | ✅ | ✅ | ✅ |
| SageMaker | ✅ | ✅ | ✅ |
| Bedrock (Claude) | ✅ | ✅ | ❌ |

**Example**: You want to fine-tune Llama 3 70B on 8× H100. `p5.48xlarge` is not available in `ap-south-1` (Mumbai), so you must train in `us-east-1` or `eu-west-1` and deploy inference in `ap-south-1`.

**Check before designing**:
- AWS: [EC2 Instance Types by Region](https://aws.amazon.com/ec2/instance-types/)
- GCP: [GPU regions](https://cloud.google.com/compute/docs/gpus)
- Azure: [Products by region](https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/)

### 4. Cost Differences

The same resource can cost very differently across regions.

| Resource | us-east-1 (cheap) | eu-west-1 (moderate) | ap-south-1 (varies) |
|---|---|---|---|
| p5.48xlarge (8× H100) | ~$100/hr | ~$110/hr | Not available |
| g5.xlarge (1× A10G) | ~$1.01/hr | ~$1.11/hr | ~$0.93/hr |
| S3 Standard (1 TB/mo) | ~$23/mo | ~$25/mo | ~$25/mo |
| Data transfer out (1 TB) | ~$90 | ~$90 | ~$140 |

**Example**: Training a large model for 1,000 hours costs ~$100,000 in `us-east-1` but the same training isn't even possible in `ap-south-1` (no H100s). You train in `us-east-1` and pay ~$0.09/GB to transfer the model artifact to `ap-south-1` for inference.

### 5. Data Transfer Costs

Data moved between regions (or out to internet) is not free.

| Transfer Direction | Cost (approx) |
|---|---|
| Within same region, same AZ | Free |
| Between AZs (same region) | ~$0.01/GB |
| Between regions | ~$0.02-0.09/GB |
| Internet egress | ~$0.05-0.12/GB |

**Example**: If you train in `us-east-1` but your dataset lives in `eu-west-1`, transferring 50 TB of data costs ~$2,000-4,000. Solution: copy the dataset to `us-east-1` before training, or use S3 Cross-Region Replication in advance.

---

## Availability Zones (AZs)

Each region has 2-6 availability zones. Each AZ is a physically separate data center with independent power, cooling, and networking. AZs are connected via high-bandwidth, low-latency fiber (typically <2ms round-trip).

### How AZs Work

```
Region: us-east-1
  ├── us-east-1a (AZ A) — Data center in northern Virginia
  ├── us-east-1b (AZ B) — Data center in different location (same metro)
  ├── us-east-1c (AZ C)
  └── us-east-1f (AZ F)

Each AZ has its own:
  - Power grid connection (separate substations)
  - Cooling infrastructure (chillers, cooling towers)
  - Network connectivity (separate fiber paths)
  - Physical security (guards, fences, access control)
```

### Why Multi-AZ Matters

A single data center can fail due to:
- Power outage (transformer explosion, grid failure)
- Cooling failure (high temperatures shut down servers)
- Fire / flooding / physical disaster
- Networking fiber cut

**Example — Single-AZ failure**:
```
Bad design: All GPUs in us-east-1a
  If AZ A goes down → entire training job lost, inference endpoint down

Good design: GPUs spread across us-east-1a, us-east-1b, us-east-1c
  If AZ A goes down → training pauses, load balancer routes to B and C
```

### AZ Names Are Not Consistent Across Accounts

`us-east-1a` in your account might map to a different physical data center than `us-east-1a` in another account. This prevents all accounts hammering the same AZ. Never hardcode AZ names.

---

## For AI Engineers: Practical Guidance

### Training Strategy

| Goal | Approach | Example |
|---|---|---|
| Minimize cost | Train in cheapest region with GPU availability | Train 70B model in `us-east-1` (~$100/hr for 8× H100) |
| Minimize latency | Deploy inference closest to users | Serve Indian users from `ap-south-1` |
| Compliance | Keep data and training within required borders | EU data → `eu-central-1` only |
| High availability | Deploy inference across 3 AZs | ALB routes across `us-east-1a`, `b`, `c` |
| Large dataset | Keep data in same region as compute | If dataset is in `eu-west-1`, train there too (avoid transfer costs) |

### Common Pitfalls

1. **GPU doesn't exist in desired region**: You designed for `ap-south-1` but p5 (H100) isn't there. Check instance availability before committing to a region.
2. **Cross-region data transfer surprise**: Training in `us-east-1` with data in `eu-west-1` incurs transfer costs. Replicate data upfront.
3. **All GPUs in one AZ**: A single AZ outage stops your entire training run. Use checkpointing and multi-AZ node groups.
4. **Region not compliant**: Storing EU user data in `us-east-1` violates GDPR. Know your compliance requirements.

### Real-World Scenario

**Building a chatbot for EU customers:**

```
Data storage:     eu-central-1 (Frankfurt)  — GDPR compliant
Training:         eu-central-1 (Frankfurt)  — p4d available (A100)
Fine-tuning:      eu-west-1 (Ireland)       — cheaper GPU spot pricing
Inference:        eu-west-1 (Ireland)       — closest to users
  Cross-region:  Copy trained model artifact from eu-central-1 to eu-west-1
                  Cost: ~$0.02/GB × 150 GB model = $3 one-time

High availability: Deploy endpoint in eu-west-1a, b, c (3 AZs)
  If one AZ fails → traffic routed to remaining AZs
```

### Region Cheatsheet

| Use Case | Recommended Region(s) | Why |
|---|---|---|
| Cheapest GPU training | `us-east-1` (N. Virginia) | Lowest GPU pricing, widest instance selection |
| EU users / GDPR | `eu-west-1` (Ireland) or `eu-central-1` (Frankfurt) | Compliant, good GPU selection |
| Asia users | `ap-southeast-1` (Singapore) or `ap-northeast-1` (Tokyo) | Low latency for Asia |
| India users | `ap-south-1` (Mumbai) | Low latency, limited GPU types |
| Australia users | `ap-southeast-2` (Sydney) | Only Australia region |
| Multi-region DR | Two regions 500+ km apart | Must survive a regional disaster (not just AZ failure) |
