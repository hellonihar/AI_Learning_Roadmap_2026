# Handwritten Digit Recognition — Results

## Performance

| Classifier | Features | Test Accuracy |
|------------|----------|--------------|
| SVC (RBF)  | HOG (324-d) | **96.8%** |
| Random Forest (200 trees) | Raw pixels (784-d) | 93.5% |

## What Was Learned

- HOG features capture edge orientations — far more robust than raw pixels for digit shapes
- SVC with RBF kernel handles the non-linear decision boundaries between similar digits well
- Random Forest on raw pixels underperforms because it treats each pixel independently and overfits to stroke thickness

## Failure Cases

- **4 ↔ 9** confusion: closed-loop vs open-loop top region is ambiguous
- **7 ↔ 1** confusion when 7 lacks a crossbar
- **3 ↔ 8** confusion when the upper loop is nearly closed

### Confusion Matrix Insight

The top-3 errors account for ~40% of all misclassifications. A secondary classifier specialised for these pairs could further improve accuracy.
