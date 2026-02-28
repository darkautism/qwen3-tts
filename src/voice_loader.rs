use anyhow::{Context, Result};
use std::path::Path;

/// Load a voice profile file, returning (flat_codec_tokens_i64, optional_ref_text).
/// Supports .json (ref_text + codec_tokens), .npy, and .pt formats.
/// Tokens are returned in [n_tokens, 16] row-major order (flattened).
pub fn load_voice_file(path: &str) -> Result<(Vec<i64>, Option<String>)> {
    let p = Path::new(path);
    match p.extension().and_then(|e| e.to_str()) {
        Some("json") => {
            let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let ref_text = data["ref_text"].as_str().map(|s| s.to_string());
            let codes: Vec<Vec<i64>> = serde_json::from_value(data["codec_tokens"].clone())
                .context("Invalid codec_tokens in voice JSON")?;
            let flat: Vec<i64> = codes.into_iter().flatten().collect();
            Ok((flat, ref_text))
        }
        Some("npy") => {
            let arr: ndarray::Array2<i64> =
                ndarray_npy::read_npy(p).with_context(|| format!("Load {}", p.display()))?;
            // Auto-transpose if saved as [16, N] instead of [N, 16]
            let arr = if arr.nrows() == 16 && arr.ncols() > 16 {
                arr.t().to_owned()
            } else {
                arr
            };
            let flat: Vec<i64> = arr.iter().copied().collect();
            Ok((flat, None))
        }
        Some("pt") => {
            use candle_core::pickle::PthTensors;
            let tensors = PthTensors::new(p, None)?;
            let t = if let Some(t) = tensors.get("")? {
                t
            } else if let Some(t) = tensors.get("codec_tokens")? {
                t
            } else if let Some(t) = tensors.get("ref_codec_tokens")? {
                t
            } else {
                anyhow::bail!("No tensor found in .pt file");
            };
            let t = t.to_dtype(candle_core::DType::I64)?;
            let shape = t.dims();
            let data = t.flatten_all()?.to_vec1::<i64>()?;
            // Auto-transpose if [16, N]
            if shape.len() == 2 && shape[0] == 16 && shape[1] > 16 {
                let n = shape[1];
                let mut transposed = vec![0i64; data.len()];
                for i in 0..16 {
                    for j in 0..n {
                        transposed[j * 16 + i] = data[i * n + j];
                    }
                }
                Ok((transposed, None))
            } else {
                Ok((data, None))
            }
        }
        _ => anyhow::bail!("Unsupported voice format: {}", path),
    }
}
