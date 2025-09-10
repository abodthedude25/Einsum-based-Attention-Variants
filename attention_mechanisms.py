import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

class MultiHeadAttention(nn.Module):
    num_heads: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, S, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        # 1. Project to Q, K, V
        qkv_proj = nn.Dense(features=self.hidden_dim * 3, name="qkv_projection")
        qkv = qkv_proj(x)
        qkv = qkv.reshape(B, S, 3, self.num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] # Shapes: (B, S, H, D)

        # Transpose to (B, H, S, D) for attention calculation
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        # 2. Calculate attention scores with einsum
        # q: (B,H,S,D), k: (B,H,S,D) -> scores: (B,H,S,S)
        scores = jnp.einsum('BHSD,BHKD->BHSK', q, k) / jnp.sqrt(head_dim)
        probs = nn.softmax(scores, axis=-1)

        # 3. Apply attention to values with einsum
        # probs: (B,H,S,K), v: (B,H,K,D) -> output: (B,H,S,D)
        output = jnp.einsum('BHSK,BHKD->BHSD', probs, v)

        # 4. Reshape and final projection
        output = output.transpose((0, 2, 1, 3)).reshape(B, S, self.hidden_dim)
        out_proj = nn.Dense(features=self.hidden_dim, name="output_projection")
        return out_proj(output)

class GroupedQueryAttention(nn.Module):
    num_q_heads: int
    num_kv_heads: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, S, _ = x.shape
        head_dim = self.hidden_dim // self.num_q_heads
        
        # Ensure Q heads are a multiple of KV heads
        assert self.num_q_heads % self.num_kv_heads == 0
        num_queries_per_kv = self.num_q_heads // self.num_kv_heads

        # Projections
        q_proj = nn.Dense(features=self.hidden_dim, name="q_projection")
        k_proj = nn.Dense(features=self.num_kv_heads * head_dim, name="k_projection")
        v_proj = nn.Dense(features=self.num_kv_heads * head_dim, name="v_projection")
        
        q = q_proj(x).reshape(B, S, self.num_q_heads, head_dim)
        k = k_proj(x).reshape(B, S, self.num_kv_heads, head_dim)
        v = v_proj(x).reshape(B, S, self.num_kv_heads, head_dim)
        
        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        # Repeat K and V heads to match Q heads
        # This is a key step in GQA
        k = jnp.repeat(k, num_queries_per_kv, axis=1)
        v = jnp.repeat(v, num_queries_per_kv, axis=1)

        # Attention calculation is now identical to MHA
        scores = jnp.einsum('BHSD,BHKD->BHSK', q, k) / jnp.sqrt(head_dim)
        probs = nn.softmax(scores, axis=-1)
        output = jnp.einsum('BHSK,BHKD->BHSD', probs, v)
        
        output = output.transpose((0, 2, 1, 3)).reshape(B, S, self.hidden_dim)
        out_proj = nn.Dense(features=self.hidden_dim, name="output_projection")
        return out_proj(output)


class MultiValueAttention(nn.Module):
    # As inspired by MatX's blog post
    num_q_heads: int
    num_k_heads: int # Usually 1 for MQA-like keys
    num_v_heads: int # Usually same as num_q_heads
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, S, _ = x.shape
        q_head_dim = self.hidden_dim // self.num_q_heads
        # For simplicity, assume all head dims are the same
        kv_head_dim = q_head_dim 
        
        # Projections
        q_proj = nn.Dense(features=self.hidden_dim, name="q_projection")
        k_proj = nn.Dense(features=self.num_k_heads * kv_head_dim, name="k_projection")
        v_proj = nn.Dense(features=self.num_v_heads * kv_head_dim, name="v_projection")

        q = q_proj(x).reshape(B, S, self.num_q_heads, q_head_dim).transpose((0, 2, 1, 3))
        k = k_proj(x).reshape(B, S, self.num_k_heads, kv_head_dim).transpose((0, 2, 1, 3))
        v = v_proj(x).reshape(B, S, self.num_v_heads, kv_head_dim).transpose((0, 2, 1, 3))

        # This einsum performs broadcasting. K has num_k_heads (G) and Q,V have num_q_heads (H).
        # We need to align them. We'll use a simpler einsum for this toy example.
        # Assuming num_k_heads=1 (MQA style)
        if self.num_k_heads == 1:
            # k shape: (B, 1, S, D) -> scores: (B, H, S, S)
            # JAX einsum will broadcast the K head dimension against the Q head dimension
            scores = jnp.einsum('BHSD,BKSD->BHSK', q, k) / jnp.sqrt(q_head_dim)
        else:
            # For a more general case, you might need grouping like GQA
            raise NotImplementedError("MVA with more than 1 K head requires grouping.")

        probs = nn.softmax(scores, axis=-1)

        # Assuming num_v_heads == num_q_heads
        # probs: (B,H,S,K), v: (B,H,K,D) -> output: (B,H,S,D)
        output = jnp.einsum('BHSK,BHKD->BHSD', probs, v)

        output = output.transpose((0, 2, 1, 3)).reshape(B, S, self.hidden_dim)
        out_proj = nn.Dense(features=self.hidden_dim, name="output_projection")
        return out_proj(output)