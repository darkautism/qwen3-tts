#!/usr/bin/env python3
"""Convert extracted talker to GGUF format for llama.cpp.

Requires llama.cpp's convert tools. Install:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make -j
"""
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Qwen3 model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file")
    parser.add_argument("--quantize", default="q4_k_m", help="Quantization type")
    parser.add_argument("--llama-cpp", default=None, help="Path to llama.cpp directory")
    args = parser.parse_args()

    # Find llama.cpp tools
    llama_dir = args.llama_cpp
    if llama_dir is None:
        candidates = [
            os.path.expanduser("~/llama.cpp"),
            "/root/llama.cpp",
            "/opt/llama.cpp",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "convert_hf_to_gguf.py")):
                llama_dir = c
                break

    if llama_dir is None:
        print("Error: llama.cpp not found. Specify --llama-cpp or install to ~/llama.cpp")
        sys.exit(1)

    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")

    # Step 1: Convert to FP16 GGUF
    fp16_path = args.output.replace(".gguf", "_f16.gguf")
    print(f"Converting to GGUF FP16: {fp16_path}")
    subprocess.run([
        sys.executable, convert_script,
        args.input,
        "--outfile", fp16_path,
        "--outtype", "f16",
    ], check=True)

    # Step 2: Quantize
    if args.quantize != "f16" and os.path.exists(quantize_bin):
        print(f"Quantizing to {args.quantize}: {args.output}")
        subprocess.run([
            quantize_bin,
            fp16_path,
            args.output,
            args.quantize.upper(),
        ], check=True)
        os.remove(fp16_path)
    else:
        os.rename(fp16_path, args.output)

    print(f"GGUF model saved: {args.output} ({os.path.getsize(args.output) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
