# Average Loss Over Dataset

## Why Average?

We don't care about loss on a single point — we care about **average loss across the entire dataset**. This tells us how the model performs overall.

```
dataset_loss = average(loss₁, loss₂, ..., lossₙ)
```

## How It's Computed

1. Forward pass through entire dataset (or batch)
2. Compute loss for each sample
3. Average all losses together
4. One number that summarizes total model performance

## Batch vs Full Dataset

In practice, computing loss over the entire dataset is too expensive (millions of samples). Instead, we use **mini-batches**.

```
Full dataset loss = average over all N samples (accurate but expensive)
Mini-batch loss  = average over 32-512 samples (noisy but cheap)
```

The mini-batch average is an **estimate** of the full dataset average. Over many batches, these estimates average out to the true loss.

## Examples

1. **Training a neural network**: You have 1M images. Computing loss over all 1M would take too long. Instead, you compute loss on batches of 256 images. Each batch gives a noisy estimate of the true loss. After enough batches, the average batch loss converges to the true dataset loss.
2. **Model comparison**: Model A has average loss 0.35, Model B has 0.42. On average, Model A makes more confident correct predictions. The average loss gives you a single number to compare models.
3. **Plateau detection**: During training, if average validation loss stops decreasing for 5 epochs, you can early-stop. You're monitoring the average loss on the validation set, not individual sample losses.
