import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        # Debug: Print the full command so you see exactly what's passing to the binary
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '0',

        # === THE SAMPLERS ===
        '--min-p', str(args.min_p),
        '--temp', str(args.temperature),

        # === THE TRINITY OF PENALTIES ===
        '--repeat-penalty', str(args.repeat_penalty),
        '--presence-penalty', str(args.presence_penalty),
        '--frequency-penalty', str(args.frequency_penalty),

        # Context settings
        '-c', str(args.ctx_size),
    ]

    if args.conversation:
        command.append("-cnv")

    run_command(command)

def signal_handler(sig, frame):
    print("\nCtrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Run BitNet Inference with Penalties')

    # Core Args
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Enable chat mode")

    # Performance
    parser.add_argument("-t", "--threads", type=int, help="Number of threads", required=False, default=8)
    parser.add_argument("-n", "--n-predict", type=int, help="Max tokens to predict (-1 = infinity)", required=False, default=-1)
    parser.add_argument("-c", "--ctx-size", type=int, help="Context size", required=False, default=4096)

    # Samplers
    parser.add_argument("--min-p", "-min-p", type=float, help="Min-P Sampling", required=False, default=0.0521)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature", required=False, default=0.9639)
    
    # === THE PENALTIES ===
    parser.add_argument("--repeat-penalty", type=float, help="Penalize exact repeats (1.0=Off, 1.1=Standard)", required=False, default=1.0421)
    parser.add_argument("--presence-penalty", type=float, help="Penalize based on existence (0.0=Off)", required=False, default=1.0621)
    parser.add_argument("--frequency-penalty", type=float, help="Penalize based on frequency (0.0=Off)", required=False, default=1.2142)

    args = parser.parse_args()
    run_inference()
