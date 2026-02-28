#!/usr/bin/env python3
"""Export code predictor weights and ONNX model."""
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

    # Get code predictor module
    if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'code_predictor'):
        cp = model.tts_model.code_predictor
    elif hasattr(model, 'code_predictor'):
        cp = model.code_predictor
    else:
        raise RuntimeError("Cannot find code_predictor in model")

    # Export weights as npz
    weights = {}

    # Codec embeddings and lm_heads (15 groups)
    for i in range(15):
        if hasattr(cp, 'codec_embeddings'):
            weights[f"codec_emb_{i}"] = cp.codec_embeddings[i].weight.detach().cpu().numpy()
        if hasattr(cp, 'lm_heads'):
            weights[f"lm_head_{i}"] = cp.lm_heads[i].weight.detach().cpu().numpy()

    # Transformer layers (5 layers)
    transformer = cp.model if hasattr(cp, 'model') else cp
    for i, layer in enumerate(transformer.layers):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, name)
            weights[f"layer_{i}_{name}"] = proj.weight.detach().cpu().numpy()

        # QK norms
        if hasattr(layer.self_attn, 'q_norm'):
            weights[f"layer_{i}_q_norm"] = layer.self_attn.q_norm.weight.detach().cpu().numpy()
            weights[f"layer_{i}_k_norm"] = layer.self_attn.k_norm.weight.detach().cpu().numpy()

        # Layer norms
        weights[f"layer_{i}_input_ln"] = layer.input_layernorm.weight.detach().cpu().numpy()
        weights[f"layer_{i}_post_ln"] = layer.post_attention_layernorm.weight.detach().cpu().numpy()

        # MLP
        weights[f"layer_{i}_gate_proj"] = layer.mlp.gate_proj.weight.detach().cpu().numpy()
        weights[f"layer_{i}_up_proj"] = layer.mlp.up_proj.weight.detach().cpu().numpy()
        weights[f"layer_{i}_down_proj"] = layer.mlp.down_proj.weight.detach().cpu().numpy()

    # Final norm
    if hasattr(transformer, 'norm'):
        weights["final_norm"] = transformer.norm.weight.detach().cpu().numpy()

    npz_path = os.path.join(args.output, "code_predictor_weights.npz")
    np.savez(npz_path, **weights)
    print(f"Weights saved: {npz_path} ({os.path.getsize(npz_path) / 1e6:.0f} MB)")

    # Export ONNX model (decode step)
    print("Exporting ONNX model...")
    try:
        export_onnx(cp, transformer, args.output)
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Numpy backend will be used as fallback.")

def export_onnx(cp, transformer, output_dir):
    """Export the transformer core as ONNX with KV-cache support."""
    import torch
    import torch.nn as nn

    num_layers = len(transformer.layers)
    hidden_size = transformer.layers[0].self_attn.q_proj.weight.shape[1]
    num_kv_heads = 8
    head_dim = hidden_size // 16

    class CodePredictorStep(nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, hidden, position, *past_kvs):
            kv_list = []
            for i in range(0, len(past_kvs), 2):
                kv_list.append((past_kvs[i], past_kvs[i+1]))

            # Run through transformer layers
            x = hidden
            new_kvs = []
            for i, layer in enumerate(self.transformer.layers):
                x, new_k, new_v = self._layer_forward(layer, x, position, kv_list[i] if i < len(kv_list) else None)
                new_kvs.extend([new_k, new_v])

            if hasattr(self.transformer, 'norm'):
                x = self.transformer.norm(x)

            return (x, *new_kvs)

        def _layer_forward(self, layer, x, positions, past_kv):
            residual = x
            x = layer.input_layernorm(x)
            x, new_k, new_v = self._attention(layer.self_attn, x, positions, past_kv)
            x = residual + x

            residual = x
            x = layer.post_attention_layernorm(x)
            x = layer.mlp(x)
            x = residual + x

            return x, new_k, new_v

        def _attention(self, attn, x, positions, past_kv):
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            v = attn.v_proj(x)

            B, S, _ = q.shape
            q = q.view(B, S, 16, head_dim).transpose(1, 2)
            k = k.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(B, S, num_kv_heads, head_dim).transpose(1, 2)

            if hasattr(attn, 'q_norm'):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

            # GQA: expand KV heads
            k_exp = k.repeat_interleave(2, dim=1)
            v_exp = v.repeat_interleave(2, dim=1)

            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v_exp)

            out = out.transpose(1, 2).reshape(B, S, -1)
            out = attn.o_proj(out)

            return out, k, v

    step_model = CodePredictorStep(transformer)
    step_model.eval()

    # Dummy inputs
    dummy_hidden = torch.randn(1, 1, hidden_size)
    dummy_position = torch.tensor([0], dtype=torch.long)
    dummy_kvs = []
    for _ in range(num_layers):
        dummy_kvs.append(torch.zeros(1, num_kv_heads, 0, head_dim))
        dummy_kvs.append(torch.zeros(1, num_kv_heads, 0, head_dim))

    input_names = ["hidden", "position"]
    for i in range(num_layers):
        input_names.extend([f"past_k_{i}", f"past_v_{i}"])

    output_names = ["output"]
    for i in range(num_layers):
        output_names.extend([f"present_k_{i}", f"present_v_{i}"])

    onnx_path = os.path.join(output_dir, "code_predictor_decode_step.onnx")

    torch.onnx.export(
        step_model,
        (dummy_hidden, dummy_position, *dummy_kvs),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "hidden": {1: "seq_len"},
            "position": {0: "seq_len"},
            **{f"past_k_{i}": {2: "past_len"} for i in range(num_layers)},
            **{f"past_v_{i}": {2: "past_len"} for i in range(num_layers)},
            **{f"present_k_{i}": {2: "total_len"} for i in range(num_layers)},
            **{f"present_v_{i}": {2: "total_len"} for i in range(num_layers)},
        },
        opset_version=17,
    )

    print(f"ONNX model saved: {onnx_path}")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        model = onnx.load(onnx_path)
        model_sim, check = simplify(model)
        if check:
            onnx.save(model_sim, onnx_path)
            print("ONNX model simplified")
    except ImportError:
        print("onnxsim not available, skipping simplification")


if __name__ == "__main__":
    main()
