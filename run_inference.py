import os
import sys
import signal
import platform
import argparse
import subprocess
import numpy as np
from bitnet_dithering_python import BitNetOrderedDithering, BitNetDitheringConfig

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    # Initialize dithering if enabled
    dithering = None
    if args.enable_dithering:
        print("🔧 Initializing BitNet ordered dithering for resolution enhancement...")
        config = BitNetDitheringConfig()
        config.enable_dithering = True
        config.dithering_strength = args.dithering_strength
        config.resolution_enhancement = args.resolution_enhancement
        config.adaptive_strength = args.adaptive_dithering
        config.bayer_matrix_size = args.bayer_matrix_size
        
        dithering = BitNetOrderedDithering()
        dithering.set_config(config)
        
        print(f"✅ Dithering configured - Strength: {config.dithering_strength}, "
              f"Resolution Enhancement: {config.resolution_enhancement}")
    
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
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        '-b', '1'
    ]
    if args.conversation:
        command.append("-cnv")
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # Usage: python run_inference.py -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington."
    parser = argparse.ArgumentParser(description='Run inference with optional ordered dithering for resolution enhancement')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict when generating text", required=False, default=-1)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=4096)
    parser.add_argument("-b", "--B", type=int, help="Size of the prompt context", required=False)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature, a hyperparameter that controls the randomness of the generated text", required=False, default=0.8)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Whether to enable chat mode or not (for instruct models.)")
    
    # Dithering-specific arguments
    parser.add_argument("--enable-dithering", action='store_true', help="Enable ordered dithering for resolution enhancement")
    parser.add_argument("--dithering-strength", type=float, help="Dithering strength (0.0-1.0)", required=False, default=0.1)
    parser.add_argument("--resolution-enhancement", action='store_true', help="Enable resolution enhancement dithering", default=True)
    parser.add_argument("--adaptive-dithering", action='store_true', help="Use adaptive dithering based on content complexity", default=True)
    parser.add_argument("--bayer-matrix-size", type=int, help="Bayer matrix size (4 or 8)", choices=[4, 8], required=False, default=4)

    args = parser.parse_args()
    run_inference()
