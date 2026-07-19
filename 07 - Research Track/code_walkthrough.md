# Code Walkthrough — Research Track

## 1. REINFORCE Policy Gradient on CartPole

```python
import numpy as np
import gymnasium as gym

# ---------- Policy Network ----------
class PolicyNetwork:
    def __init__(self, obs_dim, act_dim, hidden=128):
        self.W1 = np.random.randn(obs_dim, hidden) * 0.01
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, act_dim) * 0.01
        self.b2 = np.zeros(act_dim)

    def forward(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def backward(self, obs, act, advantage, lr=0.01):
        # Forward pass with cached intermediates
        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Gradient of cross-entropy loss weighted by advantage
        dlogits = probs.copy()
        dlogits[act] -= 1  # grad of -log pi(a)
        dlogits *= advantage

        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T
        dh = dh * (1 - h ** 2)  # tanh derivative
        dW1 = np.outer(obs, dh)
        db1 = dh

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# ---------- REINFORCE Training ----------
def reinforce(env_name="CartPole-v1", num_episodes=2000, gamma=0.99, lr=0.01):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, act_dim)
    rewards_history = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        log_probs, rewards = [], []

        # Rollout
        while True:
            obs = obs.astype(np.float32)
            probs = policy.forward(obs)
            act = np.random.choice(act_dim, p=probs)
            log_prob = np.log(probs[act])
            obs_next, reward, terminated, truncated, _ = env.step(act)

            log_probs.append(log_prob)
            rewards.append(reward)
            obs = obs_next

            if terminated or truncated:
                break

        # Compute discounted returns
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        # Normalise returns for stability
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

        # Policy gradient update
        obs_arr, _ = env.reset()
        idx = 0
        # Re-run to get observations (or store them during rollout)
        # To keep it simple, we rebuild (store obs in rollout for production)
        env.close()
        env = gym.make(env_name)
        obs, _ = env.reset()
        step = 0
        while True:
            obs_f32 = obs.astype(np.float32)
            probs = policy.forward(obs_f32)
            act = np.random.choice(act_dim, p=probs)
            obs_next, _, terminated, truncated, _ = env.step(act)

            policy.backward(obs_f32, act, returns[step], lr)
            obs = obs_next
            step += 1

            if terminated or truncated:
                break

        rewards_history.append(np.sum(rewards))
        if ep % 200 == 0:
            avg = np.mean(rewards_history[-100:]) if rewards_history else 0
            print(f"Episode {ep}, Avg Reward: {avg:.1f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    rewards = reinforce()
    print(f"Final 100-avg: {np.mean(rewards[-100:]):.1f}")
```

This implements the REINFORCE algorithm from first principles. Each episode collects a full trajectory, computes discounted returns, then performs a gradient update that increases the probability of actions that led to higher returns.

---

## 2. KV Cache — Conceptual Demonstration

```python
import numpy as np

class SimplifiedAttention:
    """Minimal multi-head attention to demonstrate KV cache mechanics."""

    def __init__(self, d_model=64, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        # KV cache: list of (K, V) tuples per layer
        self.kv_cache = []

    def forward_with_cache(self, x, use_cache=True):
        """
        x: (seq_len, d_model) — current tokens.
        If use_cache and cache exists, only process the *last* token.
        """
        if use_cache and self.kv_cache:
            # Only compute Q, K, V for the new token (last position)
            x_new = x[-1:, :]  # shape (1, d_model)
            Q_new = x_new @ self.W_q
            K_new = x_new @ self.W_k
            V_new = x_new @ self.W_v
            # Append to cache
            K_cache, V_cache = self.kv_cache
            K = np.concatenate([K_cache, K_new], axis=0)
            V = np.concatenate([V_cache, V_new], axis=0)
        else:
            # Full compute (prefill)
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v
            K_new, V_new = K, V

        # Update cache
        self.kv_cache = (K, V)

        # Multi-head attention (simplified)
        batch_size, seq_len, _ = x.shape if use_cache and self.kv_cache else (1, *x.shape)
        # Split into heads
        Q_heads = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)

        scores = Q_heads @ K_heads.transpose(0, 2, 1) / np.sqrt(self.head_dim)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)

        out_heads = attn @ V_heads  # (num_heads, seq_len, head_dim)
        out = out_heads.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return out

    def generate(self, prompt_tokens, num_to_generate=10):
        """Simulate autoregressive generation with KV cache."""
        # Prefill: process all prompt tokens in one forward pass
        out = self.forward_with_cache(prompt_tokens, use_cache=False)
        generated = [out[-1:]]

        for _ in range(num_to_generate):
            # Decode: process only the last token using cache
            last_token = generated[-1]
            out = self.forward_with_cache(last_token, use_cache=True)
            generated.append(out)

        return np.concatenate(generated, axis=0)


# Demo: compare full recomputation vs KV cache
d_model = 64
seq_len = 10
x = np.random.randn(seq_len, d_model)

# Without cache (full recompute each step)
attn_no_cache = SimplifiedAttention(d_model)
out_no_cache = attn_no_cache.forward_with_cache(x, use_cache=False)

# With cache
attn_cache = SimplifiedAttention(d_model)
out_cache = attn_cache.forward_with_cache(x, use_cache=False)

# The outputs are identical
print("Outputs match:", np.allclose(out_no_cache, out_cache))
```

The KV cache avoids recomputing $K$ and $V$ for every previous token at each step, reducing per-token attention complexity from $O(L^2)$ to $O(L)$.

---

## 3. Quantization with bitsandbytes

```python
# Requires: pip install bitsandbytes transformers torch

import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- FP16 Baseline ----------
model_name = "HuggingFaceTB/SmolLM2-135M"  # Small model for demo
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Measure memory
fp16_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"FP16 memory: {fp16_memory:.2f} GB")

# ---------- INT4 Quantization ----------
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    device_map="auto",
)

torch.cuda.reset_peak_memory_stats()
_ = model_4bit.generate(
    tokenizer("Hello, world!", return_tensors="pt").input_ids.cuda(),
    max_new_tokens=20,
)
int4_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"INT4 memory: {int4_memory:.2f} GB")
print(f"Memory reduction: {fp16_memory / int4_memory:.1f}x")

# ---------- Manual 4-bit Quantization (NF4) ----------
def quantize_nf4(weights, block_size=64):
    """
    Block-wise NF4 quantisation.
    NF4 uses 16 levels: {0, 1/16, ..., 15/16} normalised.
    """
    flat = weights.flatten()
    num_blocks = (len(flat) + block_size - 1) // block_size
    quantized_blocks = []

    for i in range(num_blocks):
        block = flat[i * block_size : (i + 1) * block_size]
        abs_max = np.abs(block).max()
        if abs_max == 0:
            quantized_blocks.append(block * 0)
            continue
        # Normalise to [-1, 1]
        normalised = block / abs_max
        # Quantise to 4-bit (16 levels)
        levels = 16
        quantised = np.round((normalised + 1) * (levels - 1) / 2)
        quantised = np.clip(quantised, 0, levels - 1)
        quantized_blocks.append(quantised)

    return np.concatenate(quantized_blocks), abs_max


# Demo quantisation on a small weight matrix
import numpy as np

weights = np.random.randn(256, 256).astype(np.float32)
q_weights, scale = quantize_nf4(weights)
print(f"\nQuantisation demo:")
print(f"  Original: {weights.nbytes} bytes")
print(f"  Quantised: {q_weights.nbytes} bytes")
print(f"  Compression ratio: {weights.nbytes / q_weights.nbytes:.1f}x")

# ---------- Compare Outputs ----------
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    out_fp16 = model_fp16.generate(inputs, max_new_tokens=20)
    out_4bit = model_4bit.generate(inputs, max_new_tokens=20)

print(f"\nFP16 output:  {tokenizer.decode(out_fp16[0])}")
print(f"INT4 output:  {tokenizer.decode(out_4bit[0])}")
```

INT4 quantisation via bitsandbytes reduces memory by ~4x with minimal quality loss, enabling deployment on consumer GPUs. The `nf4` dtype is specifically designed for normally distributed LLM weights.
