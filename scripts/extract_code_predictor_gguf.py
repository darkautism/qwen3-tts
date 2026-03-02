#!/usr/bin/env python3
"""Extract code-predictor-only GGUF from a full qwen3-tts GGUF model.

Strips talker and spk_enc tensors, keeping only code_pred + codec_embd.
Reduces model from ~1.2GB to ~170-206MB, dramatically improving cache
performance on memory-constrained devices (41% faster code prediction).

Usage:
    python extract_code_predictor_gguf.py input.gguf output.gguf
"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType, GGMLQuantizationType

QUANT_INFO = {
    GGMLQuantizationType.Q4_0: (32, 18),
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q8_0: (32, 34),
    GGMLQuantizationType.Q8_1: (32, 36),
}


def reshape_tensor_data(tensor):
    """Reshape raw bytes to numpy array preserving GGUF shape for the writer."""
    data = tensor.data.tobytes()
    gguf_shape = [int(d) for d in tensor.shape]
    mem_shape = list(reversed(gguf_shape))
    qtype = tensor.tensor_type

    if qtype == GGMLQuantizationType.F32:
        return np.frombuffer(data, dtype=np.float32).reshape(mem_shape)
    elif qtype == GGMLQuantizationType.F16:
        return np.frombuffer(data, dtype=np.float16).reshape(mem_shape)
    elif qtype in QUANT_INFO:
        block_size, type_size = QUANT_INFO[qtype]
        if len(mem_shape) == 1:
            return np.frombuffer(data, dtype=np.uint8)
        innermost = mem_shape[-1]
        bytes_per_row = (innermost // block_size) * type_size
        outer = mem_shape[:-1]
        return np.frombuffer(data, dtype=np.uint8).reshape(outer + [bytes_per_row])
    else:
        raise ValueError(f"Unsupported quant type: {qtype}")


def extract(src_path, dst_path):
    r = GGUFReader(src_path)
    w = GGUFWriter(dst_path, "qwen3-tts")

    for field_name, field in r.fields.items():
        if field_name.startswith("GGUF."):
            continue
        t = field.types[0] if len(field.types) > 0 else None
        try:
            if t == GGUFValueType.UINT32:
                w.add_uint32(field_name, int(field.parts[-1][0]))
            elif t == GGUFValueType.INT32:
                w.add_int32(field_name, int(field.parts[-1][0]))
            elif t == GGUFValueType.FLOAT32:
                w.add_float32(field_name, float(field.parts[-1][0]))
            elif t == GGUFValueType.STRING:
                w.add_string(field_name, str(bytes(field.parts[-1]), encoding="utf-8"))
            elif t == GGUFValueType.BOOL:
                w.add_bool(field_name, bool(field.parts[-1][0]))
            elif t == GGUFValueType.UINT64:
                w.add_uint64(field_name, int(field.parts[-1][0]))
        except Exception as e:
            print(f"  Skip metadata {field_name}: {e}")

    kept = 0
    for tensor in r.tensors:
        name = tensor.name
        if "code_pred" not in name and name != "talker.codec_embd.weight":
            continue
        arr = reshape_tensor_data(tensor)
        w.add_tensor(name, arr, raw_dtype=tensor.tensor_type)
        kept += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    import os
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"Extracted {kept} tensors → {dst_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.gguf>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
