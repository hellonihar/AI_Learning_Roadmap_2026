# Visualization with Matplotlib & Seaborn

## Matplotlib Basics
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="train", linewidth=2)
plt.scatter(x, y, alpha=0.5)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("My Plot")
plt.legend()
plt.grid(True)
plt.show()
```

## Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(x, y)
axes[0, 1].hist(data, bins=30)
axes[1, 0].scatter(x, y)
axes[1, 1].imshow(image)
plt.tight_layout()
```

## Seaborn
```python
import seaborn as sns

sns.histplot(df["value"], bins=30, kde=True)
sns.boxplot(x="category", y="value", data=df)
sns.scatterplot(x="feature1", y="feature2", hue="label", data=df)
sns.pairplot(df, hue="target")
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
sns.set_theme(style="whitegrid")
```
