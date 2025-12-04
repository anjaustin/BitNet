import os
import sys
import signal
import platform
import argparse
import subprocess
import tempfile

# Presets discovered through Mirostat parameter exploration
PRESETS = {
    "philosopher": {  # Cites real sources, commits to positions
        "temperature": 0.4376,
        "mirostat": 2,
        "mirostat_ent": 1.448,
        "mirostat_lr": 0.0733,
        "min_p": 0.05,
        "repeat_penalty": 1.29,
        "presence_penalty": 1.29,
        "frequency_penalty": 1.29
    },
    "3spresso": {  # PhD student on third espresso - high temp lucid dream
        "temperature": 74,
        "mirostat": 2,
        "mirostat_ent": 1.448,
        "mirostat_lr": 0.0733,
        "min_p": 0.05,
        "repeat_penalty": 1.29,
        "presence_penalty": 1.29,
        "frequency_penalty": 1.29
    },
    "Freedom90": {  # A friend.
        "temperature": 1.031,
        "mirostat": 2,
        "mirostat_ent": 1.448,
        "mirostat_lr": 0.0733,
        "min_p": 0.0521,
        "repeat_penalty": 1.29,
        "presence_penalty": 1.29,
        "frequency_penalty": 1.29
    }
}

def format_chat_prompt(system_prompt: str, user_message: str) -> str:
    """Format prompt using LLaMA 3 chat template for BitNet-b1.58-2B-4T.
    Note: llama-cli adds BOS token automatically, so we don't include <|begin_of_text|>
    """
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    # Read user message from stdin if available
    user_message = ""
    if not sys.stdin.isatty():
        user_message = sys.stdin.read().strip()
    
    # Format prompt with chat template if we have a user message
    if user_message:
        prompt = format_chat_prompt(args.prompt, user_message)
    else:
        prompt = args.prompt
    
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    
    # Write prompt to temp file to preserve special tokens
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    try:
        command = [
            f'{main_path}',
            '-m', args.model,
            '-n', str(args.n_predict),
            '-t', str(args.threads),
            '-f', prompt_file,
            '-ngl', '0',
            '-c', str(args.ctx_size),
            '--temp', str(args.temperature),
            '--min-p', str(args.min_p),
            '--repeat-penalty', str(args.repeat_penalty),
            '--presence-penalty', str(args.presence_penalty),
            '--frequency-penalty', str(args.frequency_penalty),
        ]
        if args.mirostat > 0:
            command.extend(['--mirostat', str(args.mirostat)])
            command.extend(['--mirostat-ent', str(args.mirostat_ent)])
            command.extend(['--mirostat-lr', str(args.mirostat_lr)])
        run_command(command)
    finally:
        os.unlink(prompt_file)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Run BitNet inference with LLaMA 3 chat template')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", required=False, default=-1)
    parser.add_argument("-p", "--prompt", type=str, help="System prompt (user message via stdin)", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=8)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=4096)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature", required=False, default=0.4376)
    parser.add_argument("--mirostat", type=int, help="Mirostat mode (0=off, 1=v1, 2=v2)", required=False, default=2)
    parser.add_argument("--mirostat-ent", type=float, help="Mirostat target entropy", required=False, default=1.448)
    parser.add_argument("--mirostat-lr", type=float, help="Mirostat learning rate", required=False, default=0.0733)
    parser.add_argument("--min-p", type=float, help="Min-p sampling threshold", required=False, default=0.05)
    parser.add_argument("--repeat-penalty", type=float, help="Repeat penalty", required=False, default=1.29)
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty", required=False, default=1.29)
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty", required=False, default=1.29)
    parser.add_argument("--preset", type=str, help="Parameter preset (philosopher, 3spresso)", required=False, choices=list(PRESETS.keys()))

    args = parser.parse_args()
    
    # Apply preset if specified (overrides defaults, but CLI args override preset)
    if args.preset:
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            arg_key = key.replace('_', '-')
            # Only apply preset value if user didn't explicitly set it
            if f'--{arg_key}' not in sys.argv and f'-temp' not in sys.argv:
                setattr(args, key.replace('-', '_'), value)
    
    run_inference()
