# Scaling Machine Learning

When datasets grow beyond what fits in memory or models take too long to train, you need scaling strategies.

## Training on Large Datasets

### The Problem

- 10M rows × 1,000 features = 80 GB (in float32) — doesn't fit in most machines' RAM
- Training XGBoost on 1M rows takes hours; on 100M rows, it may not converge at all
- Neural networks on TB-scale datasets need distributed training

### Strategies

| Strategy | How It Works | When to Use |
|---|---|---|
| **Out-of-core learning** | Stream data from disk in chunks, never loading all at once | Single machine, large dataset |
| **Subsampling** | Train on a representative subset | When accuracy doesn't degrade significantly |
| **Mini-batch training** | Process batches (32-1024 samples) iteratively | Neural networks, SGD-based models |
| **Distributed training** | Parallelize across multiple machines | Very large data, deep learning |

## Incremental / Online Learning

Update the model **one batch at a time** without retraining from scratch.

### How It Works

```
model = SGDClassifier()
for batch in data_stream:
    model.partial_fit(batch)  # updates weights incrementally
```

### Supported Models

- **SGD-based**: SGDClassifier, SGDRegressor (scikit-learn)
- **Naive Bayes**: MultinomialNB, BernoulliNB support `partial_fit`
- **Neural networks**: All deep learning frameworks (PyTorch, TensorFlow) train incrementally by default
- **Passive-Aggressive**: Online learning algorithm for classification/regression
- **River/Mlxtend**: Libraries designed specifically for online learning

### When to Use Online Learning

- **Data arrives as a stream** (sensor data, click streams, stock prices)
- **Concept drift** — patterns change over time, model must adapt
- **Data is too large to fit in memory**
- **Latency-sensitive** — need real-time model updates

### Challenges

- **Catastrophic forgetting**: New data overwrites old patterns. Mitigation: replay old data, use regularization.
- **Concept drift detection**: When to update? Fixed schedule? Only when performance drops?
- **Order sensitivity**: If data arrives in a biased order (all class A first, then class B), the model initially performs poorly.

**Examples:**
1. **Ad click prediction**: 1M new ad impressions per minute. An online model updates weights in real-time, adapting to changing user behavior and new ads. Batch retraining would take too long.
2. **IoT sensor monitoring**: Temperature sensors on factory equipment stream readings 24/7. An online model detects anomalies in real-time and adapts to seasonal drift (summer vs winter baselines) without manual retraining.
3. **News recommendation**: User interests change with current events. An online model updates user embeddings as new interactions arrive. A user who just read 5 politics articles starts seeing more politics — no retraining needed.

## Distributed Training (Conceptual)

Spread training across multiple machines to handle data or models that are too large for one machine.

### Data Parallelism

Split the data across machines. Each machine has a copy of the model and trains on its data shard. Gradients are synchronized across machines.

```
Machine 1: data[0:100K] → gradients →
Machine 2: data[100K:200K] → gradients →  Sync → Update model → repeat
Machine 3: data[200K:300K] → gradients →
```

- **Synchronous SGD**: Wait for all machines to finish each batch. Guarantees convergence but slow workers block everyone.
- **Asynchronous SGD**: Machines update independently. Faster but can have stale gradients.

### Model Parallelism

Split the model across machines. Each machine holds a portion of the layers/parameters.

Used when the model itself doesn't fit on one GPU (e.g., GPT-3 with 175B parameters).

### When You Need Distributed Training

| Scenario | Approach |
|---|---|
| Dataset > 1 TB | Data parallelism |
| Model > 1 GPU memory | Model parallelism |
| Need to train in minutes, not days | Data parallelism (100+ GPUs) |
| Single machine training > 1 week | Data parallelism |

**Examples:**
1. **GPT-4 training**: Estimated 10K+ GPUs in parallel. Data parallelism across GPUs (model parallelism also used). Training would take decades on a single GPU — distributed training makes it feasible in months.
2. **Image classification on ImageNet**: 1.2M images, 1,000 classes. Distributed data parallelism across 8 GPUs reduces ResNet-50 training from 2 weeks to 1 hour.
3. **Recommendation system with 1B users**: Embedding tables for 1B users don't fit on one machine. Model parallelism: user embeddings are sharded across machines. Training happens in parallel with synchronized gradient updates.

## Memory vs Compute Trade-offs

| Approach | Memory Usage | Compute | Latency |
|---|---|---|---|
| **Full batch training** | High (all data in memory) | Medium | High (epoch-level updates) |
| **Mini-batch training** | Low (one batch in memory) | High (many iterations) | Low (per-batch updates) |
| **Online learning** | Very low (one sample) | Medium | Very low (real-time) |
| **Distributed training** | Depends on shard size | Very high (multiple machines) | Medium (sync overhead) |
| **Model quantization** | Low (8-bit vs 32-bit) | Lower (integer math) | Lower |
| **Model pruning** | Low (fewer parameters) | Lower | Lower |

### Practical Guidelines

- **< 1 GB**: Standard scikit-learn / pandas workflow. Everything fits in memory.
- **1-100 GB**: Use chunking, out-of-core learning (XGBoost with `tree_method='hist'`), or subsampling.
- **100 GB - 1 TB**: Distributed computing (Spark ML, Dask, Ray). Consider online learning.
- **> 1 TB**: Distributed deep learning, data parallelism across clusters.

**Example**: An ad-tech company processes 500 GB of clickstream data daily. They use Dask for feature engineering (distributed pandas), LightGBM with histogram-based tree building (memory-efficient), and update the model incrementally every hour using the streaming data. Full retraining happens once per week on weekends.
