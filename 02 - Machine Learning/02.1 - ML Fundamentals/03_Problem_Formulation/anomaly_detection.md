# Anomaly / Outlier Detection

Identify data points that are **unusual** — they deviate significantly from the norm.

## Approaches

### 1. Statistical / Density-Based
Fit a distribution to normal data. Points with very low probability are anomalies.

### 2. Distance-Based
Points far from all neighbors are anomalies (e.g., k-NN distance outlier).

### 3. Isolation-Based
Isolation Forest: anomaly points are easier to "isolate" with random splits than normal points.

### 4. Autoencoder-Based (Deep Learning)
Train an autoencoder on normal data. At inference, high reconstruction error = anomaly (the model doesn't know how to reconstruct this unusual input).

## Examples

1. **Credit card fraud**: A user's typical spending pattern: $5–$50 at grocery stores, under $500 at electronics stores. Suddenly: $3,200 international wire transfer at 3 AM. Statistical anomaly → flagged for verification. (Uses density estimation)
2. **Manufacturing quality control**: Sensors measure vibration, temperature, and pressure on an assembly line. A bearing starting to fail produces unusual vibration patterns slightly outside normal range — detected before visible damage occurs. (Uses distance or isolation-based)
3. **Network intrusion detection**: Normal network traffic has predictable patterns (typical ports, packet sizes, connection durations). A DDoS attack or data exfiltration produces anomalous patterns — many connections to unusual ports or large outgoing data transfers at odd hours. (Uses autoencoder or statistical)

## Key Challenge

**Imbalanced data**: Anomalies are rare (e.g., 0.01% of transactions are fraud). A model that predicts "not fraud" for every input gets 99.99% accuracy but is useless. Must use specialized techniques: anomaly-specific metrics (precision@k), resampling, or one-class classification.
