# Product Image Classification — Results

## Performance

| Classifier | Features | Test Accuracy | Macro F1 |
|------------|----------|--------------|----------|
| Logistic Regression | Color hist + GLCM (36-d) | **83.2%** | 0.81 |
| Logistic Regression | Raw pixels (784-d) | 81.0% | 0.79 |
| Random Forest | Raw pixels (784-d) | 84.5% | 0.83 |

## What Was Learned

- A compact 36-d feature set matches or beats raw pixel Logistic Regression
- GLCM texture features help disambiguate coats vs dresses (different fabric patterns)
- Random Forest still wins on raw pixels because it captures non-linear pixel interactions that LR misses

## Failure Cases

- **Shirt ↔ T-shirt**: most confused pair (~18% error); features too coarse to separate collar/cut differences
- **Coat vs Pullover**: sleeve length is a key differentiator not captured by global histograms
- Low-contrast images flatten histogram and reduce discrimination power

## Next Improvements

- Add HOG features for shape information (expect +3–4% accuracy)
- Use stratified sampling if class imbalance is present
