import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        # Print the command for debugging
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
        # FIX 1: Pass the actual min_p value, not ctx_size
        # FIX 2: Use standard llama-cli flag '--min-p' (hyphen)
        '--min-p', str(args.min_p),
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
    ]
    if args.conversation:
        command.append("-cnv")
    run_command(command)

def signal_handler(sig, frame):
    print("\nCtrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", required=False, default=-1)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=8)
    # FIX 3: Clean up definition. Now accepts --min-p (preferred) or -min-p
    parser.add_argument("--min-p", "-min-p", type=float, help="Set min_p sampling", required=False, default=0.0521)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=4096)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature", required=False, default=144)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Enable chat mode")

    args = parser.parse_args()
    run_inference()
