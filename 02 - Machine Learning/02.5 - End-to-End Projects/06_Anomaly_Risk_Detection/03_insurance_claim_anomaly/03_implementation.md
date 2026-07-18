# Insurance Claim Anomaly — Implementation

## Approaches

1. **Statistical Methods** — Z-score on claim amount per procedure, MAD (median abs deviation) for robustness
2. **Isolation Forest Baseline** — sklearn implementation with tuned contamination
3. **Autoencoder** — PyTorch dense autoencoder, reconstruction MSE as anomaly score
4. **Ensemble** — Weighted combination of all three scores

## Key Code

```python
# Autoencoder for reconstruction-based anomaly detection
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train on normal claims only
model = AnomalyAutoencoder(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    recon = model(X_train_tensor)
    loss = criterion(recon, X_train_tensor)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

```python
# Score computation
model.eval()
with torch.no_grad():
    recon = model(X_test_tensor)
    reconstruction_error = ((recon - X_test_tensor) ** 2).mean(dim=1).numpy()

# Isolation Forest
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
iso_score = -iso.score_samples(X_test_scaled)

# Statistical: Modified Z-score per procedure group
from scipy.stats import median_abs_deviation
claim_amount = claims['claim_amount'].values
median = np.median(claim_amount)
mad = median_abs_deviation(claim_amount)
modified_z = 0.6745 * (claim_amount - median) / (mad + 1e-9)
```

```python
# Ensemble: rank-average the three scores
from scipy.stats import rankdata

scores = np.column_stack([
    rankdata(reconstruction_error),
    rankdata(iso_score),
    rankdata(modified_z)
])
ensemble_score = scores.mean(axis=1)
```
