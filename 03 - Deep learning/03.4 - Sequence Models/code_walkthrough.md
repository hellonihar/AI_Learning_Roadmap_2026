# Code Walkthrough: RNN and LSTM from Scratch with NumPy

This walkthrough implements a vanilla RNN cell, an LSTM cell, and a training loop for sine wave prediction using **only NumPy**. No deep learning frameworks.

```python
import numpy as np
```

## Vanilla RNN Cell

```python
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        self.x = x
        self.h_prev = h_prev
        self.h = np.tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        return self.h

    def backward(self, dh, clip_val=5.0):
        dtanh = dh * (1 - self.h ** 2)
        dW_xh = dtanh @ self.x.T
        dW_hh = dtanh @ self.h_prev.T
        db_h = np.sum(dtanh, axis=1, keepdims=True)
        dh_prev = self.W_hh.T @ dtanh
        for d in [dW_xh, dW_hh, db_h, dh_prev]:
            np.clip(d, -clip_val, clip_val, out=d)
        return dW_xh, dW_hh, db_h, dh_prev
```

### Forward Pass Explanation

The forward method computes:
$$
\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)
$$

The backward method computes the gradient of $\tanh$:
$$
\frac{\partial \tanh(a)}{\partial a} = 1 - \tanh^2(a)
$$

And propagates $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$ backward through the cell.

## LSTM Cell

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        scale = 0.01
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_f = np.zeros((hidden_size, 1))
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_i = np.zeros((hidden_size, 1))
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_c = np.zeros((hidden_size, 1))
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.b_o = np.zeros((hidden_size, 1))
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        concat = np.vstack([h_prev, x])

        self.f = self._sigmoid(self.W_f @ concat + self.b_f)
        self.i = self._sigmoid(self.W_i @ concat + self.b_i)
        self.c_tilde = np.tanh(self.W_c @ concat + self.b_c)
        self.c = self.f * self.c_prev + self.i * self.c_tilde
        self.o = self._sigmoid(self.W_o @ concat + self.b_o)
        self.h = self.o * np.tanh(self.c)
        return self.h, self.c

    def backward(self, dh, dc_next, clip_val=5.0):
        do = dh * np.tanh(self.c)
        do = do * self.o * (1 - self.o)
        dc = dh * self.o * (1 - np.tanh(self.c) ** 2) + dc_next
        dc_tilde = dc * self.i * (1 - self.c_tilde ** 2)
        di = dc * self.c_tilde * self.i * (1 - self.i)
        df = dc * self.c_prev * self.f * (1 - self.f)
        dc_prev = dc * self.f

        concat = np.vstack([self.h_prev, self.x])
        dW_f = df @ concat.T
        db_f = np.sum(df, axis=1, keepdims=True)
        dW_i = di @ concat.T
        db_i = np.sum(di, axis=1, keepdims=True)
        dW_c = dc_tilde @ concat.T
        db_c = np.sum(dc_tilde, axis=1, keepdims=True)
        dW_o = do @ concat.T
        db_o = np.sum(do, axis=1, keepdims=True)

        dh_prev = (self.W_f[:, :self.hidden_size].T @ df
                   + self.W_i[:, :self.hidden_size].T @ di
                   + self.W_c[:, :self.hidden_size].T @ dc_tilde
                   + self.W_o[:, :self.hidden_size].T @ do)

        for d in [dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o, dh_prev, dc_prev]:
            np.clip(d, -clip_val, clip_val, out=d)
        return dW_f, dW_i, dW_c, dW_o, db_f, db_i, db_c, db_o, dh_prev, dc_prev

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
```

### LSTM Forward Gates

The LSTM forward pass computes the four gates as described in the LSTM section. The concatenated input `[h_prev; x]` is multiplied by each gate's weight matrix.

## Sine Wave Prediction: Training Loop

```python
def generate_sine_data(seq_length, num_samples):
    X, y = [], []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        t = np.linspace(start, start + seq_length, seq_length + 1)
        y_seq = np.sin(t)
        X.append(y_seq[:-1].reshape(1, seq_length))
        y.append(y_seq[1:].reshape(1, seq_length))
    return np.array(X), np.array(y)
```

```python
def train_rnn(hidden_size=64, seq_length=20, epochs=100, lr=0.01):
    X, y = generate_sine_data(seq_length, 1000)
    cell = RNNCell(1, hidden_size)
    W_hy = np.random.randn(1, hidden_size) * 0.01
    b_y = np.zeros((1, 1))

    for epoch in range(epochs):
        total_loss = 0
        dW_xh, dW_hh, db_h = [np.zeros_like(p) for p in [cell.W_xh, cell.W_hh, cell.b_h]]
        dW_hy, db_y = np.zeros_like(W_hy), np.zeros_like(b_y)

        for i in range(len(X)):
            h = np.zeros((hidden_size, 1))
            grads = [np.zeros_like(p) for p in [cell.W_xh, cell.W_hh, cell.b_h]]
            gW_hy, gb_y = np.zeros_like(W_hy), np.zeros_like(b_y)

            # Forward through time
            hs = []
            for t in range(seq_length):
                x_t = X[i, :, t:t+1].T
                h = cell.forward(x_t, h)
                hs.append(h)

            # Compute loss and backward through time
            dh_next = np.zeros((hidden_size, 1))
            for t in reversed(range(seq_length)):
                y_pred = W_hy @ hs[t] + b_y
                dy = y_pred - y[i, :, t:t+1].T
                total_loss += 0.5 * np.sum(dy ** 2)
                dW_hy += dy @ hs[t].T
                db_y += dy
                dh = W_hy.T @ dy + dh_next
                gW_xh, gW_hh, gb_h, dh_next = cell.backward(dh)

            # Accumulate
            for j, g in enumerate([gW_xh, gW_hh, gb_h]):
                [dW_xh, dW_hh, db_h][j] += g

        # Update
        for param, grad in zip([cell.W_xh, cell.W_hh, cell.b_h, W_hy, b_y],
                                [dW_xh, dW_hh, db_h, dW_hy, db_y]):
            param -= lr * grad / len(X)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(X):.6f}")

    print("Training complete!")
    return cell, W_hy, b_y
```

### Training Details

- **Data**: Sine waves sliced into overlapping windows. Input is $\sin(t)$, target is $\sin(t+1)$.
- **BPTT**: The loss gradient flows backward through all time steps. At each step, the hidden state gradient `dh_next` is passed to the previous step.
- **Gradient clipping**: Prevents exploding gradients (default clip value of 5.0).
- **Loss**: Mean squared error between predicted and actual next time step.

## Prediction

```python
def predict(cell, W_hy, b_y, seed, steps=50):
    h = np.zeros((cell.W_hh.shape[0], 1))
    output = []
    x = seed.reshape(-1, 1)
    for _ in range(steps):
        h = cell.forward(x, h)
        y_pred = W_hy @ h + b_y
        output.append(y_pred.item())
        x = y_pred
    return np.array(output)
```

## Summary

This code demonstrates:
1. A **vanilla RNN cell** with forward and backward (BPTT) passes.
2. An **LSTM cell** with full gating mechanism and gradient computation.
3. **Training** on sine wave prediction using gradient descent.
4. **Inference** by autoregressive generation (feeding predictions as inputs).

All using NumPy — no PyTorch, TensorFlow, or JAX. Understanding these implementations builds intuition for how gradients flow through time and how gating mechanisms stabilize training.
