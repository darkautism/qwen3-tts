#!/usr/bin/env python3
"""Extract talker weights as standard Qwen3 format."""
import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", default="Qwen/Qwen3-TTS")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading {args.hf_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float32, trust_remote_code=True
    )

    os.makedirs(args.output, exist_ok=True)

    # Extract talker (first 28 layers of the model)
    talker_state = {}
    for name, param in model.named_parameters():
        if name.startswith("model.layers.") or name.startswith("model.norm.") or \
           name.startswith("model.embed_tokens."):
            talker_state[name] = param.data

    # Expand codec embedding to full vocab for compatibility
    if "model.embed_tokens.weight" in talker_state:
        orig = talker_state["model.embed_tokens.weight"]
        if orig.shape[0] < 151936:
            expanded = torch.zeros(151936, orig.shape[1], dtype=orig.dtype)
            expanded[:orig.shape[0]] = orig
            talker_state["model.embed_tokens.weight"] = expanded

    from safetensors.torch import save_file
    save_file(talker_state, os.path.join(args.output, "model.safetensors"))

    # Save config
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "torch_dtype": "float32",
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Talker saved to {args.output}")

if __name__ == "__main__":
    main()
