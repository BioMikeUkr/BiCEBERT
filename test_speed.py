import torch
import time
import numpy as np
from transformers import AutoTokenizer, ModernBertForMaskedLM
import gc

def generate_512_token_texts(tokenizer, batch_size=32):
    """Generate texts that are exactly 512 tokens long"""
    print("Generating 512-token texts...")
    
    # Base text to repeat
    base_text = "This is a comprehensive test sentence for benchmarking model performance with various computational workloads and linguistic patterns. "
    
    texts = []
    for i in range(batch_size):
        # Start with base text and add variation
        text = f"Sample {i}: " + base_text
        
        # Tokenize and check length
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Extend or trim to exactly 512 tokens
        if len(tokens) < 512:
            # Add more text to reach 512 tokens
            extension = " Additional content for padding purposes with meaningful text. " * 20
            extended_text = text + extension
            tokens = tokenizer.encode(extended_text, add_special_tokens=True)
            
            # Trim to exactly 512 if we overshot
            if len(tokens) > 512:
                tokens = tokens[:512]
        else:
            # Trim to exactly 512
            tokens = tokens[:512]
        
        # Convert back to text
        final_text = tokenizer.decode(tokens, skip_special_tokens=False)
        texts.append(final_text)
    
    # Verify all texts are exactly 512 tokens
    for i, text in enumerate(texts):
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        if token_count != 512:
            print(f"Warning: Text {i} has {token_count} tokens, not 512")
    
    print(f"Generated {len(texts)} texts with exactly 512 tokens each")
    return texts

def benchmark_model(model_path, model_type="auto", num_iterations=100, batch_size=32):
    print(f"\nBenchmarking {model_type} model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", add_prefix_space=True)
    
    # Load model based on type
    if model_type == "bice":
        from src import BiCEBertForMaskedLM
        model = BiCEBertForMaskedLM.from_pretrained(model_path)
        # Set eager attention
        if hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = "eager"
        encoder = model.base_model
    elif model_type == "modernbert":
        model = ModernBertForMaskedLM.from_pretrained(model_path)
        # Set eager attention
        if hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = "eager"
        encoder = model.base_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device).half().eval()  # Cast to float16
    
    print(f"Device: {device}")
    print(f"Precision: float16")
    print(f"Model config attention: {getattr(model.config, 'attn_implementation', 'default')}")
    
    # Generate exactly 512-token texts
    dummy_texts = generate_512_token_texts(tokenizer, batch_size)
    
    # Verify tokenization produces exactly 512 tokens
    sample_tokens = tokenizer(dummy_texts[0], return_tensors="pt", add_special_tokens=True)
    print(f"Sample text token count: {sample_tokens['input_ids'].shape[1]}")
    
    # Warmup runs
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            inputs = tokenizer(dummy_texts, padding=False, truncation=False, 
                             return_tensors="pt").to(device)
            outputs = encoder(**inputs)
    
    # Benchmark
    print(f"Starting benchmark: {num_iterations} iterations with batch_size={batch_size}, 512 tokens each")
    
    times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            # Tokenize texts (all should be exactly 512 tokens)
            inputs = tokenizer(dummy_texts, padding=False, truncation=False, 
                             return_tensors="pt").to(device)
            
            # Verify input shape
            if i == 0:
                print(f"Input shape: {inputs['input_ids'].shape}")
            
            # Time the forward pass
            start_time = time.time()
            outputs = encoder(**inputs)
            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 25 == 0:
                avg_time = np.mean(times[-25:])
                print(f"Iteration {i+1}: Avg time (last 25): {avg_time*1000:.2f}ms")
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate throughput
    sequences_per_second = batch_size / avg_time
    tokens_per_second = (batch_size * 512) / avg_time
    
    results = {
        "model_path": model_path,
        "model_type": model_type,
        "device": device,
        "precision": "float16",
        "batch_size": batch_size,
        "sequence_length": 512,
        "num_iterations": num_iterations,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "sequences_per_second": sequences_per_second,
        "tokens_per_second": tokens_per_second,
        "attention_type": getattr(model.config, 'attn_implementation', 'default')
    }
    
    print(f"\nResults for {model_type}:")
    print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Min/Max time: {min_time*1000:.2f} / {max_time*1000:.2f} ms")
    print(f"  Throughput: {sequences_per_second:.1f} sequences/second")
    print(f"  Token throughput: {tokens_per_second:.0f} tokens/second")
    
    # Cleanup
    del model, encoder, inputs, outputs
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()
    
    return results

def run_speed_benchmark():
    models_to_test = [
        ("BiCEbert-mlm/checkpoint-4000", "bice"),
        ("modern_bert/checkpoint-4000", "modernbert")
    ]
    
    all_results = []
    
    print("="*60)
    print("MODEL SPEED BENCHMARK")
    print("="*60)
    print(f"Test configuration:")
    print(f"  - Iterations: 100")
    print(f"  - Batch size: 32") 
    print(f"  - Sequence length: 512 tokens (exact)")
    print(f"  - Precision: float16")
    print(f"  - Attention: eager")
    
    for model_path, model_type in models_to_test:
        try:
            results = benchmark_model(model_path, model_type, num_iterations=100, batch_size=32)
            all_results.append(results)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error benchmarking {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print("\n" + "="*60)
    print("SPEED COMPARISON")
    print("="*60)
    
    if len(all_results) >= 2:
        bice_result = next((r for r in all_results if r['model_type'] == 'bice'), None)
        modernbert_result = next((r for r in all_results if r['model_type'] == 'modernbert'), None)
        
        if bice_result and modernbert_result:
            bice_time = bice_result['avg_time_ms']
            modernbert_time = modernbert_result['avg_time_ms']
            speedup = modernbert_time / bice_time  # Higher is better for BiCE
            
            print(f"BiCE model:      {bice_time:.2f} ms/batch ({bice_result['sequences_per_second']:.1f} seq/s, {bice_result['tokens_per_second']:.0f} tok/s)")
            print(f"ModernBERT:      {modernbert_time:.2f} ms/batch ({modernbert_result['sequences_per_second']:.1f} seq/s, {modernbert_result['tokens_per_second']:.0f} tok/s)")
            print(f"Speedup ratio:   {speedup:.2f}x ({'BiCE faster' if speedup > 1 else 'ModernBERT faster'})")
    
    # Save results
    import json
    from datetime import datetime
    
    output_file = f"speed_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    return all_results

if __name__ == "__main__":
    results = run_speed_benchmark()