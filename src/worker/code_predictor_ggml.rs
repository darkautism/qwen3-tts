//! GGML C code predictor via static FFI.
//! Only compiled with `ggml-backend` feature.

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};
use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::Device;

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

        // Load codec embeddings from GGUF (same tensors as Candle backend)
        let device = Device::Cpu;
        let mut file = std::fs::File::open(&gguf_path)
            .with_context(|| format!("Open {}", gguf_path.display()))?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Read GGUF: {}", e))?;

        let mut codec_embeddings = Vec::with_capacity(NUM_GROUPS);
        for i in 0..NUM_GROUPS {
            let emb = ct
                .tensor(
                    &mut file,
                    &format!("code_pred.codec_embd.{}.weight", i),
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("Load codec_embd.{}: {}", i, e))?;
            let emb_f32 = emb
                .dequantize(&device)
                .map_err(|e| anyhow::anyhow!("Dequantize codec_embd.{}: {}", i, e))?;
            let shape = emb_f32
                .dims2()
                .map_err(|e| anyhow::anyhow!("dims2 codec_embd.{}: {}", i, e))?;
            let data = emb_f32
                .to_vec2::<f32>()
                .map_err(|e| anyhow::anyhow!("to_vec2 codec_embd.{}: {}", i, e))?;
            let flat: Vec<f32> = data.into_iter().flatten().collect();
            codec_embeddings.push(Array2::from_shape_vec((shape.0, shape.1), flat)?);
        }

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

fn find_gguf(cp_dir: &Path) -> Result<std::path::PathBuf> {
    // GGML backend prefers Q4_0 (highest GGML speedup)
    // Stripped code-predictor-only GGUFs preferred (much smaller, better cache perf)
    for name in &[
        "code-predictor-q4_0.gguf",
        "code-predictor-q8_0.gguf",
        "qwen3-tts-0.6b-q4_0.gguf",
        "qwen3-tts-0.6b-q8_0.gguf",
        "qwen3-tts-0.6b-f16.gguf",
    ] {
        let p = cp_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!("No GGUF model found in {}", cp_dir.display())
}
