#!/usr/bin/env python3
"""Export vocoder to ONNX format."""
import argparse
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", default="Qwen/Qwen3-TTS")
    parser.add_argument("--output", required=True)
    parser.add_argument("--codes_length", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading {args.hf_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float32, trust_remote_code=True
    )

    os.makedirs(args.output, exist_ok=True)

    # Get vocoder/speech decoder
    if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'speech_tokenizer'):
        vocoder = model.tts_model.speech_tokenizer.decoder
    elif hasattr(model, 'speech_tokenizer'):
        vocoder = model.speech_tokenizer.decoder
    else:
        raise RuntimeError("Cannot find speech_tokenizer.decoder in model")

    vocoder.eval()

    # Trace with fixed input shape
    dummy_codes = torch.randint(0, 2048, (1, args.codes_length, 16), dtype=torch.long)

    class VocoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, audio_codes):
            return self.decoder(audio_codes)

    wrapper = VocoderWrapper(vocoder)
    wrapper.eval()

    onnx_path = os.path.join(args.output, f"vocoder_traced_{args.codes_length}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_codes,
            onnx_path,
            input_names=["audio_codes"],
            output_names=["audio"],
            opset_version=17,
        )

    print(f"Vocoder ONNX saved: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.0f} MB)")

    # Simplify
    try:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(onnx_path)
        model_sim, check = simplify(model_onnx)
        if check:
            onnx.save(model_sim, onnx_path)
            print("Vocoder ONNX simplified")
    except ImportError:
        print("onnxsim not available, skipping")


if __name__ == "__main__":
    main()
