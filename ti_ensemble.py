#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import argparse

"""
Ensemble inference: Run multiple passes at different Mirostat entropies, then synthesize.
Usage: echo "Your question" | python3 ti_ensemble.py -p "You are a friend."
"""

def format_chat_prompt(system_prompt: str, user_message: str) -> str:
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def run_inference(prompt: str, model: str, mirostat_ent: float, threads: int = 8, n_predict: int = 512) -> str:
    """Run a single inference pass and return the output text."""
    build_dir = "build"
    main_path = os.path.join(build_dir, "bin", "llama-cli")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    try:
        command = [
            main_path,
            '-m', model,
            '-n', str(n_predict),
            '-t', str(threads),
            '-f', prompt_file,
            '-ngl', '0',
            '-c', '4096',
            '--temp', '0.963',
            '--mirostat', '2',
            '--mirostat-ent', str(mirostat_ent),
            '--mirostat-lr', '0.0528',
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        # Extract just the assistant response (after the prompt echo)
        if "assistant\n" in output:
            response = output.split("assistant\n")[-1]
            # Remove the performance stats at the end
            if "llama_perf" in response:
                response = response.split("llama_perf")[0]
            if "[end of text]" in response:
                response = response.split("[end of text]")[0]
            return response.strip()
        return output
    finally:
        os.unlink(prompt_file)

def main():
    parser = argparse.ArgumentParser(description='Ensemble inference with Mirostat entropy diversity')
    parser.add_argument("-m", "--model", type=str, default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="System prompt")
    parser.add_argument("-t", "--threads", type=int, default=8)
    parser.add_argument("--ent1", type=float, default=2.618, help="First pass entropy (default: 1.618)")
    parser.add_argument("--ent2", type=float, default=6.44, help="Second pass entropy (default: 4.44)")
    parser.add_argument("--ent-synth", type=float, default=3.96, help="Synthesis pass entropy (default: 1.96)")
    parser.add_argument("-n", "--n-predict", type=int, default=-1, help="Max tokens per pass (-1 = unlimited)")
    
    args = parser.parse_args()
    
    # Read user message from stdin
    user_message = ""
    if not sys.stdin.isatty():
        user_message = sys.stdin.read().strip()
    
    if not user_message:
        print("Error: Provide user message via stdin")
        sys.exit(1)
    
    print(f"=== PASS 1: Entropy {args.ent1} (focused/academic) ===\n")
    prompt1 = format_chat_prompt(args.prompt, user_message)
    response1 = run_inference(prompt1, args.model, args.ent1, args.threads, args.n_predict)
    print(response1)
    print(f"\n{'='*60}\n")
    
    print(f"=== PASS 2: Entropy {args.ent2} (creative/exploratory) ===\n")
    prompt2 = format_chat_prompt(args.prompt, user_message)
    response2 = run_inference(prompt2, args.model, args.ent2, args.threads, args.n_predict)
    print(response2)
    print(f"\n{'='*60}\n")
    
    print(f"=== SYNTHESIS: Entropy {args.ent_synth} ===\n")
    
    # Truncate responses to ~1500 chars each to fit in context
    r1_trunc = response1[:1500] + "..." if len(response1) > 1500 else response1
    r2_trunc = response2[:1500] + "..." if len(response2) > 1500 else response2
    
    synthesis_question = f"""Original question: "{user_message}"

Perspective A: {r1_trunc}

Perspective B: {r2_trunc}

Your task: Create a NEW integrated answer that:
1. Identifies what A and B AGREE on
2. Notes where they DIFFER or contradict
3. Produces NOVEL insights that emerge from combining both
4. Do NOT simply list points from A then B - truly INTEGRATE them
5. Be concise - one cohesive response, not a comparison"""

    prompt_synth = format_chat_prompt(
        "You synthesize multiple viewpoints into unified insights. You find hidden connections and create new understanding.",
        synthesis_question
    )
    response_synth = run_inference(prompt_synth, args.model, args.ent_synth, args.threads, args.n_predict)
    print(response_synth)

if __name__ == "__main__":
    main()
