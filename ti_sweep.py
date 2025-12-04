#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

"""
Parallel Mirostat entropy sweep - run multiple entropy values simultaneously.
Usage: echo "Your question" | python3 ti_sweep.py -p "You are a friend."
"""

def format_chat_prompt(system_prompt: str, user_message: str) -> str:
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def run_single_inference(args_tuple):
    """Run a single inference pass. Returns (entropy, token_count, response_preview)."""
    prompt, model, mirostat_ent, threads = args_tuple
    build_dir = "build"
    main_path = os.path.join(build_dir, "bin", "llama-cli")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    try:
        command = [
            main_path,
            '-m', model,
            '-n', '-1',
            '-t', str(threads),
            '-f', prompt_file,
            '-ngl', '0',
            '-c', '4096',
            '--temp', '0.438',
            '--mirostat', '2',
            '--mirostat-ent', str(mirostat_ent),
            '--mirostat-lr', '0.1',
        ]
        
        start = time.time()
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        
        # Extract response and count tokens
        response = ""
        token_count = 0
        if "assistant\n" in output:
            response = output.split("assistant\n")[-1]
            if "llama_perf" in response:
                # Extract token count from perf stats
                for line in response.split('\n'):
                    if "eval time" in line and "runs" in line:
                        try:
                            token_count = int(line.split('/')[1].strip().split()[0])
                        except:
                            pass
                response = response.split("llama_perf")[0]
            if "[end of text]" in response:
                response = response.split("[end of text]")[0]
            response = response.strip()
        
        # Get first 200 chars as preview
        preview = response[:200] + "..." if len(response) > 200 else response
        
        return (mirostat_ent, token_count, elapsed, preview, response)
    except subprocess.TimeoutExpired:
        return (mirostat_ent, -1, 300, "TIMEOUT", "")
    except Exception as e:
        return (mirostat_ent, -1, 0, f"ERROR: {e}", "")
    finally:
        os.unlink(prompt_file)

def main():
    parser = argparse.ArgumentParser(description='Parallel Mirostat entropy sweep')
    parser.add_argument("-m", "--model", type=str, default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="System prompt")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Threads per inference (default 4)")
    parser.add_argument("-w", "--workers", type=int, default=3, help="Parallel workers (default 3)")
    parser.add_argument("--ent-start", type=float, default=1.0, help="Starting entropy")
    parser.add_argument("--ent-end", type=float, default=3.0, help="Ending entropy")
    parser.add_argument("--ent-step", type=float, default=0.25, help="Entropy step size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full responses")
    
    args = parser.parse_args()
    
    # Read user message from stdin
    user_message = ""
    if not sys.stdin.isatty():
        user_message = sys.stdin.read().strip()
    
    if not user_message:
        print("Error: Provide user message via stdin")
        sys.exit(1)
    
    # Generate entropy values to test
    entropies = []
    ent = args.ent_start
    while ent <= args.ent_end + 0.001:
        entropies.append(round(ent, 3))
        ent += args.ent_step
    
    print(f"=== MIROSTAT ENTROPY SWEEP ===")
    print(f"Range: {args.ent_start} to {args.ent_end}, step {args.ent_step}")
    print(f"Testing {len(entropies)} entropy values with {args.workers} parallel workers")
    print(f"Question: {user_message[:80]}...")
    print(f"{'='*60}\n")
    
    prompt = format_chat_prompt(args.prompt, user_message)
    
    # Build argument tuples for parallel execution
    work_items = [(prompt, args.model, ent, args.threads) for ent in entropies]
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_inference, item): item[2] for item in work_items}
        
        for future in as_completed(futures):
            ent, tokens, elapsed, preview, full = future.result()
            results.append((ent, tokens, elapsed, preview, full))
            print(f"  ent={ent:.3f}: {tokens:4d} tokens, {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    # Sort by entropy and display summary
    results.sort(key=lambda x: x[0])
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (sorted by entropy)")
    print(f"{'='*60}")
    print(f"{'Entropy':>8} | {'Tokens':>6} | {'Time':>6} | Preview")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*40}")
    
    for ent, tokens, elapsed, preview, _ in results:
        preview_short = preview[:40].replace('\n', ' ')
        print(f"{ent:>8.3f} | {tokens:>6d} | {elapsed:>5.1f}s | {preview_short}...")
    
    # Find peak
    peak = max(results, key=lambda x: x[1])
    print(f"\n{'='*60}")
    print(f"PEAK: entropy {peak[0]:.3f} with {peak[1]} tokens")
    print(f"Total sweep time: {total_time:.1f}s (vs ~{len(entropies) * 20:.0f}s sequential)")
    
    if args.verbose:
        print(f"\n{'='*60}")
        print("FULL RESPONSES:")
        for ent, tokens, _, _, full in results:
            print(f"\n--- Entropy {ent:.3f} ({tokens} tokens) ---")
            print(full[:1000])

if __name__ == "__main__":
    main()
