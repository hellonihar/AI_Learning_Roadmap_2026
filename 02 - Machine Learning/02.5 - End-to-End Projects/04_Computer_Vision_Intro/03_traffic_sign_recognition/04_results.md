# Traffic Sign Recognition — Results

## Performance

| Approach | Test Accuracy | Macro F1 |
|----------|--------------|----------|
| HOG + Color + PCA + SVC (no aug) | 91.2% | 0.90 |
| HOG + Color + PCA + SVC (with aug) | **93.8%** | 0.93 |
| Raw pixels + RF (baseline) | 82.5% | 0.80 |

## What Was Learned

- Affine augmentation (+2.6% accuracy) is critical — real traffic signs vary in tilt and distance
- PCA reduced 624-d → ~120-d with 0% accuracy loss, speeding up both training and inference
- `class_weight='balanced'` improved rare-class F1 by 5–8 points (e.g., 20 km/h vs 120 km/h signs)
- HOG captures shape (circle, triangle, octagon) which is the primary discriminator

## Failure Cases

- **30 km/h vs 80 km/h**: digits inside circles are small — pixel resolution limits readability
- **Stop vs No-entry**: both red and circular — color histogram alone isn't enough; need shape context
- Motion-blurred samples (simulated in test set) degrade HOG edge detection

## Insights

- The top-5 best recognised classes are unique-shape signs (Stop, Yield, Priority road)
- The bottom-5 are speed limits — consider a dedicated digit-reading sub-model for speed signs
