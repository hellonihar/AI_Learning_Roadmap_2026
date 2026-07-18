# Traffic Sign Recognition — Dataset

## Source
[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) — 50,000+ images of 43 traffic sign classes. Images are 15×15 to 250×250 pixels, resized to 32×32 for this project.

## Size
| Split | Samples |
|-------|---------|
| Train | 39,209 |
| Test  | 12,630 |

## Features
- **HOG features**: 576-d (9 orientations, 8×8 cells, 2×2 blocks, 32×32 image → 4×4×4×9)
- **Color features**: 48-d (16-bin histogram for each of R, G, B channels → 48)
- **Combined**: 624-d before PCA
- **After PCA (95% variance)**: ~120-d

## Challenges
- Real-world lighting variation (overcast, direct sun, dusk)
- Motion blur from vehicle speed
- Small signs in wide-angle frames (low resolution)
- Class imbalance — some signs have 10× fewer samples than others
