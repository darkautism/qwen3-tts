#!/usr/bin/env python3
"""
Qwen3-TTS model conversion for RK3588 distributed inference.
Produces all files needed for talker and predictor roles.

Usage:
    source .venv/bin/activate
    python3 convert_all_rk3588.py --model-dir ./external_repos/Qwen3-TTS-12Hz-0.6B-Base --output ./qwen3_tts_models
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS for RK3588")
    parser.add_argument("--model-dir", required=True, help="Path to HF model dir")
    parser.add_argument("--output", default="./qwen3_tts_models", help="Output directory")
    parser.add_argument("--vocoder-rknn", default=None, help="Path to existing vocoder.rknn")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    out = os.path.abspath(args.output)

    # Create output structure
    talker_dir = os.path.join(out, "talker")
    talker_emb_dir = os.path.join(talker_dir, "embeddings")
    pred_dir = os.path.join(out, "predictor")
    pred_emb_dir = os.path.join(pred_dir, "embeddings")
    pred_cp_dir = os.path.join(pred_dir, "code_predictor")
    for d in [talker_emb_dir, pred_emb_dir, pred_cp_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  Qwen3-TTS → RK3588 Conversion")
    print(f"  Model: {model_dir}")
    print(f"  Output: {out}")
    print("=" * 60)

    # ── Step 1: Load model ─────────────────────────────────────
    print("\n[1/6] Loading model weights...")
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    print(f"  Loaded {len(weights)} tensors")

    # ── Step 2: Extract tokenizer ──────────────────────────────
    print("\n[2/6] Extracting tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tok_out = os.path.join(talker_dir, "tokenizer.json")
    tokenizer.save_pretrained(talker_dir)
    if os.path.exists(tok_out):
        print(f"  tokenizer.json: {os.path.getsize(tok_out) / 1024:.0f} KB")
    else:
        print("  WARNING: tokenizer.json not generated, trying backend_tokenizer...")
        tokenizer.backend_tokenizer.save(tok_out)
        print(f"  tokenizer.json: {os.path.getsize(tok_out) / 1024:.0f} KB")

    # ── Step 3: Extract embeddings ─────────────────────────────
    print("\n[3/6] Extracting embeddings...")

    # Text embedding [151936, 2048]
    text_emb = weights["talker.model.text_embedding.weight"].float().cpu().numpy()
    np.save(os.path.join(talker_emb_dir, "text_embedding.npy"), text_emb)
    print(f"  text_embedding: {text_emb.shape}")

    # Codec embedding [3072, 1024] - from talker.model
    codec_emb = weights["talker.model.codec_embedding.weight"].float().cpu().numpy()
    np.save(os.path.join(talker_emb_dir, "codec_embedding.npy"), codec_emb)
    np.save(os.path.join(pred_emb_dir, "codec_embedding.npy"), codec_emb)
    print(f"  codec_embedding: {codec_emb.shape}")

    # Codec head [3072, 1024]
    codec_head = weights["talker.codec_head.weight"].float().cpu().numpy()
    np.save(os.path.join(talker_emb_dir, "codec_head.npy"), codec_head)
    print(f"  codec_head: {codec_head.shape}")

    # Text projection MLP
    proj_prefix = "talker.text_projection."
    proj_keys = [k for k in weights if k.startswith(proj_prefix)]
    print(f"  Projection keys: {proj_keys}")

    fc1_w = fc1_b = fc2_w = fc2_b = None
    for k in proj_keys:
        short = k[len(proj_prefix):].replace(".", "_")  # e.g. linear_fc1_weight
        arr = weights[k].float().cpu().numpy()
        np.save(os.path.join(talker_emb_dir, f"text_projection_{short}.npy"), arr)
        print(f"    {short}: {arr.shape}")
        if "fc1" in k and "weight" in k:
            fc1_w = arr
        elif "fc1" in k and "bias" in k:
            fc1_b = arr
        elif "fc2" in k and "weight" in k:
            fc2_w = arr
        elif "fc2" in k and "bias" in k:
            fc2_b = arr

    # Pre-compute tts_pad_embed [1024] for predictor
    TTS_PAD_TOKEN_ID = 151671
    tts_pad_raw = text_emb[TTS_PAD_TOKEN_ID]  # [2048]

    if all(x is not None for x in [fc1_w, fc1_b, fc2_w, fc2_b]):
        h = tts_pad_raw @ fc1_w.T + fc1_b
        h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
        tts_pad_embed = (h @ fc2_w.T + fc2_b).astype(np.float32)
        np.save(os.path.join(talker_emb_dir, "tts_pad_embed.npy"), tts_pad_embed)
        np.save(os.path.join(pred_emb_dir, "tts_pad_embed.npy"), tts_pad_embed)
        print(f"  tts_pad_embed: {tts_pad_embed.shape}")
    else:
        print("  WARNING: Could not compute tts_pad_embed (missing projection weights)")

    # ── Step 4: Export code predictor ──────────────────────────
    print("\n[4/6] Exporting code predictor...")
    export_code_predictor(weights, config, pred_cp_dir)

    # ── Step 5: Extract talker + GGUF ─────────────────────────
    if not args.skip_gguf:
        print("\n[5/6] Extracting talker and converting to GGUF...")
        extract_talker_and_convert_gguf(weights, config, model_dir, talker_dir)
    else:
        print("\n[5/6] Skipping GGUF (--skip-gguf)")

    # ── Step 6: Copy vocoder RKNN ─────────────────────────────
    print("\n[6/6] Vocoder RKNN...")
    voc_rknn = args.vocoder_rknn
    if voc_rknn is None:
        candidates = [
            os.path.expanduser("~/vocoder_traced_64_sim_q8.rknn"),
            os.path.expanduser("~/vocoder_traced_64_fp16.rknn"),
        ]
        voc_rknn = next((c for c in candidates if os.path.exists(c)), None)

    if voc_rknn and os.path.exists(voc_rknn):
        dst = os.path.join(pred_dir, "vocoder.rknn")
        shutil.copy2(voc_rknn, dst)
        print(f"  Copied {voc_rknn} → {dst} ({os.path.getsize(dst)/1024/1024:.0f} MB)")
    else:
        print("  WARNING: No vocoder RKNN found. Place vocoder.rknn in predictor/ manually.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Conversion complete!")
    print("=" * 60)
    print(f"\nOutput structure ({out}):")
    for root, dirs, files in os.walk(out):
        level = root.replace(out, "").count(os.sep)
        indent = "  " * level
        print(f"  {indent}{os.path.basename(root)}/")
        for f in sorted(files):
            sz = os.path.getsize(os.path.join(root, f))
            if sz > 1024 * 1024:
                print(f"  {indent}  {f} ({sz/1024/1024:.1f} MB)")
            else:
                print(f"  {indent}  {f} ({sz/1024:.0f} KB)")

    print(f"\nUpload to HuggingFace Hub:")
    print(f"  huggingface-cli upload <your-repo> {out}/ .")


def export_code_predictor(weights, config, output_dir):
    """Export code predictor ONNX + weights."""
    import torch.nn as nn

    talker_cfg = config.get("talker_config", {})
    cp_cfg = talker_cfg.get("code_predictor_config", {})

    hidden_size = cp_cfg.get("hidden_size", 1024)
    num_layers = cp_cfg.get("num_hidden_layers", 5)
    num_groups = cp_cfg.get("num_code_groups", 16) - 1  # 15 groups to predict
    vocab_size = cp_cfg.get("vocab_size", 2048)

    print(f"  hidden={hidden_size}, layers={num_layers}, groups={num_groups}")

    # Extract code predictor state dict
    prefix = "talker.code_predictor."
    cp_state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

    # Save codec embeddings and lm_heads as npz
    emb_dict = {}
    for i in range(num_groups):
        emb_key = f"model.codec_embedding.{i}.weight"
        head_key = f"lm_head.{i}.weight"
        if emb_key in cp_state:
            emb_dict[f"codec_emb_{i}"] = cp_state[emb_key].float().cpu().numpy()
        if head_key in cp_state:
            emb_dict[f"lm_head_{i}"] = cp_state[head_key].float().cpu().numpy()

    # Also save layer weights for potential head_dim detection
    for k, v in cp_state.items():
        if "q_proj" in k and "layers.0" in k:
            emb_dict["layer_0_q_proj"] = v.float().cpu().numpy()
            break

    npz_path = os.path.join(output_dir, "code_predictor_weights.npz")
    np.savez(npz_path, **emb_dict)
    print(f"  Weights: {npz_path} ({os.path.getsize(npz_path)/1024/1024:.1f} MB)")

    # Build and export ONNX model
    try:
        sys.path.insert(0, os.path.expanduser("~/external_repos/qwen3-tts-rknn-russian"))
        from qwen_tts.core.models.configuration_qwen3_tts import (
            Qwen3TTSTalkerCodePredictorConfig,
            Qwen3TTSTalkerConfig,
        )
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        )

        talker_full_cfg = Qwen3TTSTalkerConfig(**talker_cfg)
        cp_config = Qwen3TTSTalkerCodePredictorConfig(**cp_cfg)

        model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            cp_config, talker_full_cfg
        )
        model.load_state_dict(cp_state, strict=False)
        model.eval()

        # Create core wrapper
        class CodePredictorCore(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.layers = m.model.layers
                self.norm = m.model.norm
                self.rotary_emb = m.model.rotary_emb
                self.small_to_mtp_projection = m.small_to_mtp_projection

            def forward(self, hidden_states, position_ids):
                hidden_states = self.small_to_mtp_projection(hidden_states)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                for layer in self.layers:
                    out = layer(hidden_states, position_embeddings=position_embeddings)
                    hidden_states = out[0] if isinstance(out, tuple) else out
                return self.norm(hidden_states)

        core = CodePredictorCore(model)
        core.eval()

        dummy_h = torch.randn(1, 2, hidden_size)
        dummy_p = torch.arange(2).unsqueeze(0)

        with torch.no_grad():
            test_out = core(dummy_h, dummy_p)
        print(f"  Test: {dummy_h.shape} → {test_out.shape}")

        onnx_path = os.path.join(output_dir, "code_predictor_core.onnx")
        torch.onnx.export(
            core, (dummy_h, dummy_p), onnx_path,
            input_names=["hidden_states", "position_ids"],
            output_names=["output"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "output": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

        # Verify
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {
            "hidden_states": dummy_h.numpy(),
            "position_ids": dummy_p.numpy(),
        })
        diff = np.abs(test_out.numpy() - ort_out[0]).max()
        print(f"  Verify: max diff = {diff:.6f}")

    except Exception as e:
        print(f"  ERROR exporting ONNX: {e}")
        print("  Trying alternative: using existing code_predictor_core.onnx if available")

    # Save config
    cp_info = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_groups": num_groups,
        "vocab_size": vocab_size,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cp_info, f, indent=2)


def extract_talker_and_convert_gguf(weights, config, model_dir, talker_dir):
    """Extract talker as standalone Qwen3, convert to GGUF."""
    TEXT_VOCAB_SIZE = 151936

    talker_cfg = config.get("talker_config", {})
    talker_hf_dir = os.path.join(talker_dir, "_hf_talker")
    os.makedirs(talker_hf_dir, exist_ok=True)

    # Extract talker weights and rename to standard Qwen3 format
    talker_state = {}
    for k, v in weights.items():
        if k.startswith("talker.model.layers."):
            new_k = k.replace("talker.model.", "model.", 1)
            talker_state[new_k] = v
        elif k == "talker.model.codec_embedding.weight":
            # Use codec_embedding as embed_tokens (expanded to full vocab)
            codec_emb = v
            hidden_dim = codec_emb.shape[1]
            expanded = torch.zeros(TEXT_VOCAB_SIZE, hidden_dim, dtype=codec_emb.dtype)
            expanded[:codec_emb.shape[0]] = codec_emb
            talker_state["model.embed_tokens.weight"] = expanded
            print(f"  embed_tokens: expanded {list(codec_emb.shape)} → {list(expanded.shape)}")
        elif k == "talker.model.norm.weight":
            talker_state["model.norm.weight"] = v
        elif k == "talker.codec_head.weight":
            # Use codec_head as lm_head (expanded to full vocab)
            head = v
            expanded_head = torch.zeros(TEXT_VOCAB_SIZE, head.shape[1], dtype=head.dtype)
            expanded_head[:head.shape[0]] = head
            talker_state["lm_head.weight"] = expanded_head
            print(f"  lm_head: expanded {list(head.shape)} → {list(expanded_head.shape)}")

    if not talker_state:
        print("  ERROR: No talker weights found!")
        return

    print(f"  Extracted {len(talker_state)} tensors for talker")

    # Save as safetensors
    from safetensors.torch import save_file
    st_path = os.path.join(talker_hf_dir, "model.safetensors")
    save_file(talker_state, st_path)
    print(f"  Talker weights: {os.path.getsize(st_path)/1024/1024:.0f} MB")

    # Create config for standalone Qwen3
    qwen3_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": talker_cfg.get("hidden_size", 1024),
        "intermediate_size": talker_cfg.get("intermediate_size", 3072),
        "num_hidden_layers": talker_cfg.get("num_hidden_layers", 28),
        "num_attention_heads": talker_cfg.get("num_attention_heads", 16),
        "num_key_value_heads": talker_cfg.get("num_key_value_heads", 8),
        "head_dim": talker_cfg.get("head_dim", 128),
        "max_position_embeddings": talker_cfg.get("max_position_embeddings", 32768),
        "vocab_size": TEXT_VOCAB_SIZE,
        "rms_norm_eps": talker_cfg.get("rms_norm_eps", 1e-06),
        "rope_theta": talker_cfg.get("rope_theta", 1000000.0),
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "use_cache": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "torch_dtype": "float32",
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(talker_hf_dir, "config.json"), "w") as f:
        json.dump(qwen3_config, f, indent=2)

    # Try GGUF conversion
    llama_cpp_dir = None
    for path in [
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/llama.cpp-master"),
    ]:
        if os.path.exists(os.path.join(path, "convert_hf_to_gguf.py")):
            llama_cpp_dir = path
            break

    if llama_cpp_dir is None:
        print("  Downloading llama.cpp...")
        import urllib.request
        tar_path = os.path.expanduser("~/llama_cpp.tar.gz")
        urllib.request.urlretrieve(
            "https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz",
            tar_path,
        )
        subprocess.run(["tar", "xzf", tar_path, "-C", os.path.expanduser("~/")], check=True)
        os.remove(tar_path)
        llama_cpp_dir = os.path.expanduser("~/llama.cpp-master")

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        print(f"  ERROR: {convert_script} not found")
        return

    # Install gguf package
    subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "-q"], check=False)

    # Convert to GGUF (FP16 first, then quantize)
    gguf_f16 = os.path.join(talker_dir, "talker-f16.gguf")
    gguf_q8 = os.path.join(talker_dir, "talker-q8_0.gguf")

    print(f"  Converting to GGUF F16...")
    result = subprocess.run(
        [sys.executable, convert_script, talker_hf_dir, "--outfile", gguf_f16, "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  GGUF F16 conversion failed: {result.stderr[-500:]}")
        # Try with q8_0 directly
        result = subprocess.run(
            [sys.executable, convert_script, talker_hf_dir, "--outfile", gguf_q8, "--outtype", "q8_0"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  GGUF Q8_0 conversion also failed: {result.stderr[-500:]}")
            return

    if os.path.exists(gguf_f16):
        print(f"  GGUF F16: {os.path.getsize(gguf_f16)/1024/1024:.0f} MB")

        # Quantize to Q8_0
        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
        if os.path.exists(quantize_bin):
            print(f"  Quantizing to Q8_0...")
            subprocess.run([quantize_bin, gguf_f16, gguf_q8, "q8_0"], check=True)
            os.remove(gguf_f16)
        else:
            # Just rename F16 as the output
            os.rename(gguf_f16, gguf_q8)
            print("  llama-quantize not found, using F16 as output")

    if os.path.exists(gguf_q8):
        print(f"  GGUF: {gguf_q8} ({os.path.getsize(gguf_q8)/1024/1024:.0f} MB)")

    # Cleanup temp HF dir
    shutil.rmtree(talker_hf_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
