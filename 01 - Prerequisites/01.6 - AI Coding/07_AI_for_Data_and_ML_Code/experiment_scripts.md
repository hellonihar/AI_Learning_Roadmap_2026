# Experiment Scripts

Generate hyperparameter tuning and experiment management scripts.

## Prompt

```
"Write a hyperparameter sweep script using Optuna that:
- Searches learning rate (1e-5 to 1e-2, log-uniform)
- Batch size (16, 32, 64)
- Dropout rate (0.1 to 0.5)
- Number of layers (2, 3, 4)
- Uses TPE sampler
- Runs 20 trials
- Logs results to a CSV
- Prunes unpromising trials early"
```

## Example

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 2, 4)

    model = build_model(dropout=dropout, n_layers=n_layers)
    loader = DataLoader(dataset, batch_size=batch_size)

    val_loss = train_and_eval(model, loader, lr, trial=trial)
    return val_loss

study = optuna.create_study(
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=5),
    storage="sqlite:///experiments.db",
)
study.optimize(objective, n_trials=20)
```

## Experiment Prompts

```
- "Create a YAML config structure for experiment parameters"
- "Write a script to compare 3 model architectures with matched hyperparameters"
- "Generate an ablation study script (remove one feature at a time)"
- "Create a results comparison table across experiments (mean ± std over 5 seeds)"
- "Write a script to generate learning curves for each experiment"
```
