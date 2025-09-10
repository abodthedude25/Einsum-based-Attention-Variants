from dataclasses import dataclass

# We'll assume bfloat16 for KV cache (2 bytes per parameter)
BYTES_PER_PARAM = 2

@dataclass
class AttentionCost:
    """Models the cost of a single attention forward pass."""
    compute_gflops: float
    kv_cache_load_mb: float

def calculate_mha_cost(B: int, S: int, H: int, D: int) -> AttentionCost:
    """Calculates cost for Multi-Head Attention."""
    # FLOPs: 2 * (B*H*S*K*D) for QK^T and 2 * (B*H*S*K*D) for Probs*V
    # Here S=K (sequence length)
    flops = 2 * (B * H * S * S * D) + 2 * (B * H * S * S * D)
    
    # KV Cache Load: B * S * H * D for K, and same for V
    kv_bytes = B * S * H * D * 2 * BYTES_PER_PARAM
    
    return AttentionCost(
        compute_gflops=flops / 1e9,
        kv_cache_load_mb=kv_bytes / (1024**2)
    )

def calculate_gqa_cost(B: int, S: int, H_q: int, H_kv: int, D: int) -> AttentionCost:
    """Calculates cost for Grouped-Query Attention."""
    # Compute is the same as MHA because K/V are repeated to match Q
    flops = 2 * (B * H_q * S * S * D) + 2 * (B * H_q * S * S * D)
    
    # KV Cache Load: The key difference! We only load num_kv_heads.
    kv_bytes = B * S * H_kv * D * 2 * BYTES_PER_PARAM
    
    return AttentionCost(
        compute_gflops=flops / 1e9,
        kv_cache_load_mb=kv_bytes / (1024**2)
    )

def calculate_mva_cost(B: int, S: int, H_q: int, H_k: int, H_v: int, D: int) -> AttentionCost:
    """Calculates cost for Multi-Value Attention."""
    # FLOPs for QK^T uses num_k_heads, FLOPs for Probs*V uses num_v_heads
    # Assuming broadcasting for simplicity in the model
    flops_qk = 2 * (B * H_q * S * S * D) # Q is H_q, K is H_k (broadcasted)
    flops_pv = 2 * (B * H_q * S * S * D) # Probs is H_q, V is H_v (broadcasted)
    
    # KV Cache Load: K and V have different numbers of heads
    k_bytes = B * S * H_k * D * BYTES_PER_PARAM
    v_bytes = B * S * H_v * D * BYTES_PER_PARAM
    
    return AttentionCost(
        compute_gflops=(flops_qk + flops_pv) / 1e9,
        kv_cache_load_mb=(k_bytes + v_bytes) / (1024**2)
    )