#!/usr/bin/env python3
"""Convert vocoder ONNX to RKNN format for RK3588 NPU."""
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output RKNN model")
    parser.add_argument("--target", default="rk3588", choices=["rk3588", "rk3576"])
    parser.add_argument("--quantize", default="i8", choices=["i8", "fp16", "fp32"])
    args = parser.parse_args()

    from rknn.api import RKNN

    rknn = RKNN()

    # Detect codes_length from filename
    basename = os.path.basename(args.onnx)
    if "256" in basename:
        codes_length = 256
    else:
        codes_length = 64

    print(f"Converting {args.onnx} (codes_length={codes_length})")
    print(f"  Target: {args.target}, Quantization: {args.quantize}")

    rknn.config(
        target_platform=args.target,
        optimization_level=3,
    )

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    if args.quantize == "i8":
        # Generate calibration data
        print("Generating calibration data...")
        dataset_path = "/tmp/vocoder_calib_dataset.txt"
        with open(dataset_path, "w") as f:
            for i in range(20):
                npy_path = f"/tmp/vocoder_calib_{i}.npy"
                data = np.random.randint(0, 2048, (1, codes_length, 16), dtype=np.int64)
                np.save(npy_path, data)
                f.write(f"{npy_path}\n")

        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_path,
        )
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")

    ret = rknn.export_rknn(args.output)
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: {ret}")

    print(f"RKNN model saved: {args.output} ({os.path.getsize(args.output) / 1e6:.0f} MB)")

    rknn.release()


if __name__ == "__main__":
    main()
