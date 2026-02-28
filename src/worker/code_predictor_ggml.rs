//! GGML C code predictor via static FFI.
//! Only compiled with `ggml-backend` feature.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::ffi::CString;
use std::io::Read;
use std::os::raw::{c_char, c_float, c_int};
use std::path::Path;

const NUM_GROUPS: usize = 15;

extern "C" {
    fn code_pred_load(gguf_path: *const c_char, n_threads: c_int) -> *mut std::ffi::c_void;
    fn code_pred_predict(
        handle: *mut std::ffi::c_void,
        hidden: *const c_float,
        codebook_0_token: i32,
        output: *mut i32,
        temperature: c_float,
        top_k: i32,
    ) -> c_int;
    fn code_pred_free(handle: *mut std::ffi::c_void);
}

pub struct CodePredictorGgml {
    handle: *mut std::ffi::c_void,
    pub codec_embeddings: Vec<Array2<f32>>,
}

unsafe impl Send for CodePredictorGgml {}
unsafe impl Sync for CodePredictorGgml {}

impl CodePredictorGgml {
    pub fn load(cp_dir: &Path) -> Result<Self> {
        let gguf_path = find_gguf(cp_dir)?;
        let c_path = CString::new(gguf_path.to_str().unwrap())?;

        tracing::info!("Loading GGML code predictor: {}", gguf_path.display());
        let handle = unsafe { code_pred_load(c_path.as_ptr(), 4) };
        if handle.is_null() {
            anyhow::bail!(
                "Failed to load GGML code predictor from {}",
                gguf_path.display()
            );
        }

        // Load codec embeddings from npz
        let weights_path = cp_dir.join("code_predictor_weights.npz");
        let codec_embeddings = load_codec_embeddings(&weights_path)?;

        tracing::info!("GGML code predictor ready ({} groups)", NUM_GROUPS);
        Ok(Self {
            handle,
            codec_embeddings,
        })
    }

    pub fn predict(
        &mut self,
        hidden: &Array1<f32>,
        _code_0: i32,
        temperature: f32,
    ) -> Result<Vec<i32>> {
        let mut output = vec![0i32; NUM_GROUPS];
        let ret = unsafe {
            code_pred_predict(
                self.handle,
                hidden.as_ptr(),
                _code_0,
                output.as_mut_ptr(),
                temperature,
                50, // top_k
            )
        };
        if ret != 0 {
            anyhow::bail!("GGML code predictor failed with code {}", ret);
        }
        Ok(output)
    }
}

impl Drop for CodePredictorGgml {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { code_pred_free(self.handle) };
        }
    }
}

fn load_codec_embeddings(path: &Path) -> Result<Vec<Array2<f32>>> {
    let data = std::fs::read(path).with_context(|| format!("Read {}", path.display()))?;
    let mut cursor = std::io::Cursor::new(&data);

    let mut embeddings = Vec::with_capacity(NUM_GROUPS);
    let mut archive = zip::ZipArchive::new(&mut cursor).with_context(|| "Parse npz")?;

    for gi in 0..NUM_GROUPS {
        let name = format!("codec_emb_{}.npy", gi);
        let mut file = archive
            .by_name(&name)
            .with_context(|| format!("Find {} in npz", name))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let arr = parse_npy_f32_2d(&buf)?;
        embeddings.push(arr);
    }
    Ok(embeddings)
}

fn parse_npy_f32_2d(data: &[u8]) -> Result<Array2<f32>> {
    // Minimal npy parser: magic + header + data
    if &data[..6] != b"\x93NUMPY" {
        anyhow::bail!("Not a valid npy file");
    }
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = std::str::from_utf8(&data[10..10 + header_len])?;

    // Parse shape from header like "'shape': (2048, 1024)"
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;

    let data_start = 10 + header_len;
    let float_data: &[f32] = bytemuck::cast_slice(&data[data_start..]);
    Ok(Array2::from_shape_vec(
        (dims[0], dims[1]),
        float_data.to_vec(),
    )?)
}

fn find_gguf(cp_dir: &Path) -> Result<std::path::PathBuf> {
    for name in &[
        "qwen3-tts-0.6b-q8_0.gguf",
        "qwen3-tts-0.6b-q4_0.gguf",
        "qwen3-tts-0.6b-f16.gguf",
    ] {
        let p = cp_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!("No GGUF model found in {}", cp_dir.display())
}
