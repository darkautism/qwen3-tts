#!/usr/bin/env python3
"""Extract text/codec embeddings and projection weights."""
import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

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

    # Text embedding [151936, 2048]
    text_emb = model.model.embed_tokens.weight.detach().cpu().numpy()
    np.save(os.path.join(args.output, "text_embedding.npy"), text_emb)
    print(f"text_embedding: {text_emb.shape}")

    # Codec embedding [3072, 1024] - from tts_model
    if hasattr(model, 'tts_model'):
        codec_emb = model.tts_model.codec_embedding.weight.detach().cpu().numpy()
    else:
        codec_emb = model.codec_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(args.output, "codec_embedding.npy"), codec_emb)
    print(f"codec_embedding: {codec_emb.shape}")

    # Codec head [3072, 1024]
    if hasattr(model, 'tts_model'):
        codec_head = model.tts_model.codec_head.weight.detach().cpu().numpy()
    else:
        codec_head = model.codec_head.weight.detach().cpu().numpy()
    np.save(os.path.join(args.output, "codec_head.npy"), codec_head)
    print(f"codec_head: {codec_head.shape}")

    # Text projection MLP weights
    if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'text_projection'):
        proj = model.tts_model.text_projection
    elif hasattr(model, 'text_projection'):
        proj = model.text_projection
    else:
        print("Warning: text_projection not found, trying alternative paths...")
        proj = None

    if proj is not None:
        for name, param in proj.named_parameters():
            fname = f"text_projection_{name.replace('.', '_')}.npy"
            arr = param.detach().cpu().numpy()
            np.save(os.path.join(args.output, fname), arr)
            print(f"  {fname}: {arr.shape}")

    # Pre-compute tts_pad_embed [1024] for predictor worker
    TTS_PAD_TOKEN_ID = 151671
    tts_pad_raw = text_emb[TTS_PAD_TOKEN_ID]  # [2048]
    if proj is not None:
        fc1_w = proj.linear_fc1.weight.detach().cpu().numpy()  # [2048, 2048]
        fc1_b = proj.linear_fc1.bias.detach().cpu().numpy()    # [2048]
        fc2_w = proj.linear_fc2.weight.detach().cpu().numpy()  # [1024, 2048]
        fc2_b = proj.linear_fc2.bias.detach().cpu().numpy()    # [1024]
        h = tts_pad_raw @ fc1_w.T + fc1_b
        h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
        tts_pad_embed = h @ fc2_w.T + fc2_b
        np.save(os.path.join(args.output, "tts_pad_embed.npy"), tts_pad_embed.astype(np.float32))
        print(f"tts_pad_embed: {tts_pad_embed.shape}")

    print(f"\nEmbeddings saved to {args.output}")

if __name__ == "__main__":
    main()
