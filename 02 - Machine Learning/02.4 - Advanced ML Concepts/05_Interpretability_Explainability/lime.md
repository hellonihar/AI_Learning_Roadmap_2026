# LIME (Local Interpretable Model-agnostic Explanations)

Explains **individual predictions** by fitting a simple, interpretable model locally around that prediction.

*A focused deep-dive. For the broader context of interpretability, see the [Interpretability overview](interpretability.md).*

## How LIME Works

1. Take a single prediction you want to explain
2. Perturb the input — create variations by randomly turning features on/off
3. Get the model's predictions on these perturbed samples
4. Fit a **simple linear model** (or decision tree) weighted by proximity to the original input
5. The simple model's coefficients are the explanation

```
Original input: "This movie was amazing and thrilling"
Perturbed:      "This movie was amazing"       → positive (0.92)
                "This movie was thrilling"     → positive (0.88)
                "This movie was amazing not"   → negative (0.23)
                "This movie was boring"        → negative (0.05)
                
Local linear model learns:
  "amazing"  → +2.1 (pushes toward positive)
  "thrilling"→ +1.8
  "not"      → -1.5
  "boring"   → -2.3
```

## LIME vs SHAP

| Aspect | LIME | SHAP |
|---|---|---|
| **Stability** | Less stable (random perturbations vary) | Consistent (deterministic) |
| **Speed** | Fast — only needs local perturbations | Slower for exact values |
| **Global explanations** | Not designed for it | Yes (aggregate SHAP) |
| **Interpretability** | Very intuitive (linear coefficients) | Requires understanding Shapley values |
| **Model requirement** | Any model (black-box OK) | Any model (but has optimized versions) |

## When to Use LIME

- **Quick debugging** — fast explanation for a handful of predictions
- **Non-technical stakeholders** — linear coefficients are easier to explain than Shapley values
- **Text and image data** — LIME is particularly good at highlighting which words/pixels drove a prediction
- **When SHAP is too slow** — for very large models where KernelSHAP is impractical

## LIME for Text

LIME perturbs text by removing words. The linear model's coefficients show which words pushed toward or against the prediction.

**Example**: Spam classifier explanation:
```
Email: "Congratulations! You've won a FREE iPhone. Click now."
→ Spam (probability 0.97)
Top words driving "spam":
  "free"      +2.5
  "congratulations" +1.8
  "won"       +1.5
  "click"     +1.2
```

## LIME for Images

LIME perturbs image by turning super-pixels (contiguous pixel regions) on/off. The explanation highlights which regions influenced the prediction.

**Example**: Classifying an image as "dog":
```
Original image: [dog image]
Super-pixels that most support "dog":
  - Face region (+3.2)
  - Ear shape (+2.1)
  - Fur texture (+1.5)
Super-pixels that slightly oppose "dog":
  - Background grass (-0.3)
```

## Limitations

- **Instability**: Running LIME twice on the same prediction can give different explanations (random perturbations differ). Set a random seed for reproducibility.
- **Local fidelity only**: The linear approximation is only valid very close to the original point. Don't extrapolate.
- **Feature definition matters**: For tabular data, how you define "turning off" a feature (mean imputation? zero?) affects explanations.

## Examples

1. **Customer support ticket routing**: A ticket was misrouted to billing instead of tech support. LIME shows the decision was driven by words "payment" (+2.0) and "invoice" (+1.5), even though the content was about "payment API not working." The team adds a rule: if "API" + "payment" co-occur, route to tech support.
2. **Content moderation**: A flagged post is being appealed. LIME shows the model focused on the word "kill" (+3.0) in "killing it at the gym today 💪" — a false positive. The moderation team adds context-aware rules.
3. **Medical image triage**: An X-ray was flagged as "urgent." LIME highlights the specific region of the image (a super-pixel in the upper left lung area) that drove the high-risk score. The radiologist checks that region first, confirming the model's focus is clinically appropriate.
