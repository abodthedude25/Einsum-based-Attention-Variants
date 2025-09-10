import jax
import jax.numpy as jnp
from attention_mechanisms import MultiHeadAttention, GroupedQueryAttention, MultiValueAttention
from cost_model import calculate_mha_cost, calculate_gqa_cost, calculate_mva_cost

def run_analysis():
    """
    Runs a performance cost analysis on the different attention mechanisms
    and prints a comparison table.
    """
    # --- Simulation Parameters ---
    # Simulating a ~7B model profile
    BATCH_SIZE = 8
    SEQ_LEN = 4096
    HIDDEN_DIM = 4096
    NUM_HEADS = 32
    HEAD_DIM = HIDDEN_DIM // NUM_HEADS

    print("Running Attention Mechanism Cost Analysis...")
    print(f"Config: Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, HiddenDim={HIDDEN_DIM}, Heads={NUM_HEADS}\n")

    # --- Calculate Costs ---
    cost_mha = calculate_mha_cost(B=BATCH_SIZE, S=SEQ_LEN, H=NUM_HEADS, D=HEAD_DIM)
    
    # GQA with 4x reduction in KV heads
    cost_gqa = calculate_gqa_cost(B=BATCH_SIZE, S=SEQ_LEN, H_q=NUM_HEADS, H_kv=8, D=HEAD_DIM)
    
    # MVA inspired by MatX: 1 K head, but full V heads
    cost_mva = calculate_mva_cost(B=BATCH_SIZE, S=SEQ_LEN, H_q=NUM_HEADS, H_k=1, H_v=NUM_HEADS, D=HEAD_DIM)

    # --- Print Results Table ---
    baseline_kv_mb = cost_mha.kv_cache_load_mb

    header = "| Mechanism             | Compute (GFLOPs) | KV Cache Load (MB) | KV Cache Reduction |"
    separator = "| --------------------- | ---------------- | ------------------ | ------------------ |"
    
    row_mha = f"| MHA ({NUM_HEADS}H)             | {cost_mha.compute_gflops:<16.1f} | {cost_mha.kv_cache_load_mb:<18.1f} | 1.0x (Baseline)    |"
    row_gqa = f"| GQA ({NUM_HEADS}Q, 8KV)        | {cost_gqa.compute_gflops:<16.1f} | {cost_gqa.kv_cache_load_mb:<18.1f} | **{baseline_kv_mb/cost_gqa.kv_cache_load_mb:.1f}x** |"
    row_mva = f"| MVA ({NUM_HEADS}Q, 1K, {NUM_HEADS}V)    | {cost_mva.compute_gflops:<16.1f} | {cost_mva.kv_cache_load_mb:<18.1f} | **{baseline_kv_mb/cost_mva.kv_cache_load_mb:.1f}x** |"

    print(header)
    print(separator)
    print(row_mha)
    print(row_gqa)
    print(row_mva)

def test_implementations():
    """Tests that the JAX modules run without errors."""
    print("\n--- Testing JAX Implementations ---")
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((4, 64, 256)) # Smaller shapes for quick test
    
    # MHA
    mha_model = MultiHeadAttention(num_heads=8, hidden_dim=256)
    params_mha = mha_model.init(key, dummy_input)['params']
    output_mha = mha_model.apply({'params': params_mha}, dummy_input)
    print(f"MHA Output Shape: {output_mha.shape}")
    
    # GQA
    gqa_model = GroupedQueryAttention(num_q_heads=8, num_kv_heads=2, hidden_dim=256)
    params_gqa = gqa_model.init(key, dummy_input)['params']
    output_gqa = gqa_model.apply({'params': params_gqa}, dummy_input)
    print(f"GQA Output Shape: {output_gqa.shape}")

    # MVA
    mva_model = MultiValueAttention(num_q_heads=8, num_k_heads=1, num_v_heads=8, hidden_dim=256)
    params_mva = mva_model.init(key, dummy_input)['params']
    output_mva = mva_model.apply({'params': params_mva}, dummy_input)
    print(f"MVA Output Shape: {output_mva.shape}")


if __name__ == "__main__":
    run_analysis()
    test_implementations()