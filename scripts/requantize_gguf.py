#!/usr/bin/env python3
"""Re-quantize a GGUF file from Q8_0 to Q4_0/Q4_K.

Reads an existing GGUF and writes a new one with selected tensors
re-quantized to a smaller format. Useful for reducing code predictor
model size to improve inference speed on bandwidth-limited hardware.

Usage:
    python3 requantize_gguf.py input.gguf output.gguf --type q4_0
    python3 requantize_gguf.py input.gguf output.gguf --type q4_0 --prefix code_pred
"""

import argparse
import sys
import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType


def dequantize_q8_0(data_bytes: bytes, gguf_shape: tuple) -> np.ndarray:
    """Dequantize Q8_0 data to float32 in memory layout (reversed GGUF shape)."""
    n_elements = 1
    for s in gguf_shape:
        n_elements *= s

    block_size = 32
    n_blocks = n_elements // block_size

    # Q8_0 format: each block = 2 bytes (f16 scale) + 32 bytes (int8 quants) = 34 bytes
    block_bytes = 34
    expected_bytes = n_blocks * block_bytes

    if len(data_bytes) < expected_bytes:
        raise ValueError(
            f"Q8_0 data too short: {len(data_bytes)} < {expected_bytes}"
        )

    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * block_bytes
        scale = np.frombuffer(data_bytes[offset : offset + 2], dtype=np.float16)[0]
        quants = np.frombuffer(
            data_bytes[offset + 2 : offset + block_bytes], dtype=np.int8
        )
        result[i * block_size : (i + 1) * block_size] = quants.astype(np.float32) * float(scale)

    # GGUF shape is [cols, rows]; memory layout is (rows, cols)
    mem_shape = tuple(reversed(gguf_shape))
    return result.reshape(mem_shape)


def main():
    parser = argparse.ArgumentParser(description="Re-quantize GGUF model")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output GGUF file")
    parser.add_argument(
        "--type",
        choices=["q4_0", "q4_k", "q4_1"],
        default="q4_0",
        help="Target quantization type",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only re-quantize tensors with this prefix (e.g. 'code_pred')",
    )
    args = parser.parse_args()

    target_type_map = {
        "q4_0": GGMLQuantizationType.Q4_0,
        "q4_k": GGMLQuantizationType.Q4_K,
        "q4_1": GGMLQuantizationType.Q4_1,
    }
    target_type = target_type_map[args.type]

    print(f"Reading {args.input}...")
    reader = GGUFReader(args.input)

    # Copy metadata
    arch = None
    for k, v in reader.fields.items():
        if k == "general.architecture":
            arch = str(bytes(v.parts[-1]), "utf-8")
            break

    if not arch:
        arch = "qwen3-tts"

    print(f"Architecture: {arch}")
    print(f"Tensors: {len(reader.tensors)}")
    print(f"Target quantization: {args.type}")
    if args.prefix:
        print(f"Prefix filter: {args.prefix}")

    writer = GGUFWriter(args.output, arch)

    # Copy all metadata fields
    for k, v in reader.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture":
            continue
        try:
            parts = v.parts
            if v.types and len(parts) > 0:
                # Get the actual data part (last part usually)
                data = v.parts[-1]
                if hasattr(data, "__len__") and len(data) == 0:
                    continue
                # Try to reconstruct the value
                vtype = v.types[0] if v.types else None
                if vtype == gguf.GGUFValueType.STRING:
                    val = str(bytes(data), "utf-8")
                    writer.add_string(k, val)
                elif vtype == gguf.GGUFValueType.UINT32:
                    writer.add_uint32(k, int(data[0]))
                elif vtype == gguf.GGUFValueType.INT32:
                    writer.add_int32(k, int(data[0]))
                elif vtype == gguf.GGUFValueType.FLOAT32:
                    writer.add_float32(
                        k, float(np.frombuffer(bytes(data), dtype=np.float32)[0])
                    )
                elif vtype == gguf.GGUFValueType.UINT64:
                    writer.add_uint64(k, int(data[0]))
                elif vtype == gguf.GGUFValueType.BOOL:
                    writer.add_bool(k, bool(data[0]))
        except Exception as e:
            print(f"  Warning: skipping metadata '{k}': {e}")

    # Process tensors
    q8_count = 0
    converted_count = 0
    skipped_names = ["norm", "bias", "embed", "lm_head"]

    for tensor in reader.tensors:
        name = tensor.name
        ttype = int(tensor.tensor_type)
        gguf_shape = tuple(int(s) for s in tensor.shape)
        data = tensor.data.tobytes()

        should_convert = (
            ttype == int(GGMLQuantizationType.Q8_0)
            and (args.prefix is None or name.startswith(args.prefix))
            and not any(skip in name for skip in skipped_names)
        )

        if should_convert:
            q8_count += 1
            try:
                # Dequantize Q8_0 → F32 in memory layout (reversed GGUF shape)
                f32_data = dequantize_q8_0(data, gguf_shape)
                # Re-quantize to target (writer infers GGUF shape from memory layout)
                quantized = gguf.quants.quantize(f32_data, target_type)
                writer.add_tensor(name, quantized, raw_dtype=target_type)
                converted_count += 1

                if converted_count <= 3 or converted_count % 10 == 0:
                    old_mb = len(data) / 1024 / 1024
                    new_mb = quantized.nbytes / 1024 / 1024
                    print(
                        f"  Converted {name}: {gguf_shape} Q8_0({old_mb:.1f}MB) → {args.type}({new_mb:.1f}MB)"
                    )
            except Exception as e:
                print(f"  Warning: failed to convert {name}: {e}, keeping Q8_0")
                # Reshape raw data to memory layout for writer
                mem_shape = tuple(reversed(gguf_shape))
                raw = np.frombuffer(data, dtype=np.uint8)
                n_rows = mem_shape[0] if len(mem_shape) > 1 else 1
                raw = raw.reshape(n_rows, -1)
                writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            # Keep original type — reshape raw data to memory layout
            if ttype == int(GGMLQuantizationType.Q8_0):
                raw_dtype = GGMLQuantizationType.Q8_0
            elif ttype == 1:
                raw_dtype = GGMLQuantizationType.F16
            elif ttype == 0:
                raw_dtype = GGMLQuantizationType.F32
            else:
                raw_dtype = GGMLQuantizationType(ttype)

            raw = np.frombuffer(data, dtype=np.uint8)
            mem_shape = tuple(reversed(gguf_shape))
            if len(mem_shape) > 1:
                n_rows = mem_shape[0]
                raw = raw.reshape(n_rows, -1)

            writer.add_tensor(name, raw, raw_dtype=raw_dtype)

    print(f"\nConverted {converted_count}/{q8_count} Q8_0 tensors to {args.type}")
    print(f"Writing {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done!")


if __name__ == "__main__":
    main()
