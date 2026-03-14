#!/usr/bin/env python3
"""
Base chat script for testing uncompressed models.
Used to compare performance against compressed models.
"""

import torch
import argparse
import time
import psutil
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable torch compilation completely
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention which can cause issues

def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def main():
    parser = argparse.ArgumentParser(description='Base model chat for performance comparison')
    parser.add_argument('model_path', type=str, help='Path to model directory')
    parser.add_argument('--prompt', type=str, default='Write a haiku', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=20, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for generation')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path).expanduser()
    
    print(f"Loading base model from: {model_path}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Configure CPU threading for optimal performance
    if device == 'cpu':
        num_threads = torch.get_num_threads()
        print(f"CPU threads available: {num_threads}")
        # Use all available threads for multithreaded CPU inference
        torch.set_num_threads(num_threads)
    
    print(f"Using device: {device}")
    
    # Memory before loading
    memory_before = get_memory_usage_gb()
    print(f"Memory before loading: {memory_before:.2f} GB")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    load_start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map=device if device == 'cuda' else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True if device == 'cpu' else False
    )
    
    if device == 'cpu':
        model = model.to(device)
    
    load_time = time.time() - load_start
    print(f"Model loaded in: {load_time:.2f}s")
    
    # Memory after loading  
    memory_after = get_memory_usage_gb()
    memory_used = memory_after - memory_before
    print(f"Memory after loading: {memory_after:.2f} GB")
    print(f"Model memory usage: {memory_used:.2f} GB")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Generate response
    print("\nGenerating response...")
    generation_start = time.time()
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True if args.temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - generation_start
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(args.prompt):].strip()
    
    output_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
    tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
    
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Output tokens: {output_tokens}")
    print(f"Speed: {tokens_per_second:.2f} tokens/second")
    
    print(f"\n--- Generated Response ---")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {generated_text}")
    
    # Final memory check
    memory_final = get_memory_usage_gb()
    print(f"\nFinal memory usage: {memory_final:.2f} GB")
    
    # Performance summary
    print(f"\n--- Performance Summary ---")
    print(f"Model: {model_path.name}")
    print(f"Load time: {load_time:.2f}s")
    print(f"Generation time: {generation_time:.2f}s") 
    print(f"Memory usage: {memory_used:.2f} GB")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
    print(f"Device: {device}")

if __name__ == '__main__':
    main()