use anyhow::{Context, Result};
#[cfg(feature = "rknn-vocoder")]
use std::os::raw::{c_int, c_void};
use std::path::Path;

const SAMPLES_PER_TOKEN: usize = 1920;

enum VocoderBackend {
    Onnx(OnnxVocoder),
    #[cfg(feature = "rknn-vocoder")]
    Rknn(RknnVocoder),
}

/// Vocoder: converts codec tokens → audio waveform
pub struct Vocoder {
    backend: VocoderBackend,
    pub max_tokens: usize,
}

impl Vocoder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let path_str = model_path.to_str().unwrap_or("");

        #[cfg(feature = "rknn-vocoder")]
        if path_str.ends_with(".rknn") {
            let max_tokens = if path_str.contains("256") { 256 } else { 64 };
            let rknn = RknnVocoder::load(model_path)?;
            tracing::info!("RKNN INT8 vocoder ready: max_tokens={}", max_tokens);
            return Ok(Self {
                backend: VocoderBackend::Rknn(rknn),
                max_tokens,
            });
        }

        #[cfg(not(feature = "rknn-vocoder"))]
        if path_str.ends_with(".rknn") {
            anyhow::bail!(
                "RKNN vocoder requires --features rknn-vocoder. Rebuild with:\n  \
                 cargo build --release --features rknn-vocoder"
            );
        }

        if !path_str.ends_with(".onnx") {
            anyhow::bail!("Unsupported vocoder format: {}", model_path.display());
        }

        let onnx = OnnxVocoder::load(model_path)?;
        let max_tokens = onnx.max_tokens;
        tracing::info!(
            "ONNX FP32 vocoder ready: max_tokens={} (noise-free)",
            max_tokens
        );
        Ok(Self {
            backend: VocoderBackend::Onnx(onnx),
            max_tokens,
        })
    }

    /// Convert codec tokens [n_tokens, 16] → audio f32 samples
    pub fn synthesize(&mut self, codes: &[i64], n_tokens: usize) -> Result<Vec<f32>> {
        let mut audio_chunks = Vec::new();
        let mut chunk_start = 0;

        while chunk_start < n_tokens {
            let chunk_end = (chunk_start + self.max_tokens).min(n_tokens);
            let chunk_len = chunk_end - chunk_start;

            // Pad to max_tokens
            let mut padded = vec![0i64; self.max_tokens * 16];
            for i in 0..chunk_len {
                for j in 0..16 {
                    padded[i * 16 + j] = codes[(chunk_start + i) * 16 + j];
                }
            }

            let chunk_audio = match &mut self.backend {
                VocoderBackend::Onnx(onnx) => onnx.run(&padded, self.max_tokens)?,
                #[cfg(feature = "rknn-vocoder")]
                VocoderBackend::Rknn(rknn) => rknn.run(&padded, self.max_tokens)?,
            };
            let actual_samples = chunk_len * SAMPLES_PER_TOKEN;
            audio_chunks.extend_from_slice(&chunk_audio[..actual_samples.min(chunk_audio.len())]);
            chunk_start = chunk_end;
        }

        Ok(audio_chunks)
    }
}

// ============================================================
// RKNN FFI (librknnrt.so) — only with rknn-vocoder feature
// ============================================================
#[cfg(feature = "rknn-vocoder")]
#[repr(C)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: u32,
    fmt: u32,
}

#[cfg(feature = "rknn-vocoder")]
#[repr(C)]
struct RknnOutput {
    want_float: u8,
    is_prealloc: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
}

#[cfg(feature = "rknn-vocoder")]
#[repr(C)]
#[allow(dead_code)]
struct RknnInputOutputNum {
    n_input: u32,
    n_output: u32,
}

#[cfg(feature = "rknn-vocoder")]
const RKNN_TENSOR_INT64: u32 = 8;
#[cfg(feature = "rknn-vocoder")]
const RKNN_TENSOR_NHWC: u32 = 0;
#[cfg(feature = "rknn-vocoder")]
const RKNN_NPU_CORE_0_1_2: u32 = 7;

#[cfg(feature = "rknn-vocoder")]
struct RknnVocoder {
    lib: libloading::Library,
    ctx: u64,
}

#[cfg(feature = "rknn-vocoder")]
unsafe impl Send for RknnVocoder {}
#[cfg(feature = "rknn-vocoder")]
unsafe impl Sync for RknnVocoder {}

#[cfg(feature = "rknn-vocoder")]
impl RknnVocoder {
    fn load(model_path: &Path) -> Result<Self> {
        // Load librknnrt.so
        let lib_paths = [
            "/lib/librknnrt.so",
            "/usr/lib/librknnrt.so",
            "/usr/local/lib/librknnrt.so",
        ];
        let lib_path = lib_paths
            .iter()
            .find(|p| Path::new(p).exists())
            .ok_or_else(|| anyhow::anyhow!("librknnrt.so not found"))?;

        let lib = unsafe { libloading::Library::new(lib_path) }.context("Load librknnrt.so")?;

        // Read model file
        let model_data =
            std::fs::read(model_path).with_context(|| format!("Read {}", model_path.display()))?;

        // Init RKNN (pass NULL for extend)
        let mut ctx: u64 = 0;
        let ret: c_int = unsafe {
            let init: libloading::Symbol<
                unsafe extern "C" fn(*mut u64, *const c_void, u32, u32, *mut c_void) -> c_int,
            > = lib.get(b"rknn_init")?;
            init(
                &mut ctx,
                model_data.as_ptr() as *const c_void,
                model_data.len() as u32,
                0,
                std::ptr::null_mut(),
            )
        };

        if ret != 0 {
            anyhow::bail!("rknn_init failed: {}", ret);
        }

        // Set core mask to use all 3 NPU cores
        let ret: c_int = unsafe {
            let set_core: libloading::Symbol<unsafe extern "C" fn(u64, u32) -> c_int> =
                lib.get(b"rknn_set_core_mask")?;
            set_core(ctx, RKNN_NPU_CORE_0_1_2)
        };
        if ret != 0 {
            tracing::warn!(
                "rknn_set_core_mask failed: {} (continuing with default)",
                ret
            );
        }

        tracing::info!("RKNN vocoder initialized: ctx={}", ctx);
        Ok(Self { lib, ctx })
    }

    fn run(&self, codes: &[i64], max_tokens: usize) -> Result<Vec<f32>> {
        let input_size = (max_tokens * 16 * std::mem::size_of::<i64>()) as u32;
        tracing::info!(
            "RKNN run: max_tokens={}, input_size={}, codes_len={}",
            max_tokens,
            input_size,
            codes.len()
        );

        let mut input = RknnInput {
            index: 0,
            buf: codes.as_ptr() as *mut c_void,
            size: input_size,
            pass_through: 0, // Let RKNN convert INT64 → model's internal format
            type_: RKNN_TENSOR_INT64,
            fmt: RKNN_TENSOR_NHWC,
        };

        // Set inputs
        let ret: c_int = unsafe {
            let set_inputs: libloading::Symbol<
                unsafe extern "C" fn(u64, u32, *mut RknnInput) -> c_int,
            > = self.lib.get(b"rknn_inputs_set")?;
            set_inputs(self.ctx, 1, &mut input)
        };
        if ret != 0 {
            anyhow::bail!("rknn_inputs_set failed: {}", ret);
        }
        tracing::info!("RKNN inputs set OK");

        // Run
        let ret: c_int = unsafe {
            let run: libloading::Symbol<unsafe extern "C" fn(u64, *mut c_void) -> c_int> =
                self.lib.get(b"rknn_run")?;
            run(self.ctx, std::ptr::null_mut())
        };
        if ret != 0 {
            anyhow::bail!("rknn_run failed: {}", ret);
        }
        tracing::info!("RKNN run OK");

        // Get outputs
        let mut output = RknnOutput {
            want_float: 1,
            is_prealloc: 0,
            index: 0,
            buf: std::ptr::null_mut(),
            size: 0,
        };

        let ret: c_int = unsafe {
            let get_outputs: libloading::Symbol<
                unsafe extern "C" fn(u64, u32, *mut RknnOutput, *mut c_void) -> c_int,
            > = self.lib.get(b"rknn_outputs_get")?;
            get_outputs(self.ctx, 1, &mut output, std::ptr::null_mut())
        };
        if ret != 0 {
            anyhow::bail!("rknn_outputs_get failed: {}", ret);
        }

        // Copy output audio
        let n_floats = output.size as usize / std::mem::size_of::<f32>();
        let audio =
            unsafe { std::slice::from_raw_parts(output.buf as *const f32, n_floats).to_vec() };

        // Release output
        unsafe {
            let release: libloading::Symbol<
                unsafe extern "C" fn(u64, u32, *mut RknnOutput) -> c_int,
            > = self.lib.get(b"rknn_outputs_release")?;
            release(self.ctx, 1, &mut output);
        }

        Ok(audio)
    }
}

#[cfg(feature = "rknn-vocoder")]
impl Drop for RknnVocoder {
    fn drop(&mut self) {
        unsafe {
            if let Ok(destroy) = self
                .lib
                .get::<unsafe extern "C" fn(u64) -> c_int>(b"rknn_destroy")
            {
                destroy(self.ctx);
            }
        }
    }
}

// ============================================================
// ONNX FP32 Vocoder (noise-free, CPU-based) — default
// ============================================================
struct OnnxVocoder {
    session: ort::session::Session,
    max_tokens: usize,
}

impl OnnxVocoder {
    fn load(model_path: &Path) -> Result<Self> {
        let session = ort::session::Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)
            .with_context(|| format!("Load ONNX vocoder: {}", model_path.display()))?;

        // Extract max_tokens from input shape [1, max_tokens, 16]
        let max_tokens = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .and_then(|s| s.get(1).map(|&d| d as usize))
            .unwrap_or(64);

        Ok(Self {
            session,
            max_tokens,
        })
    }

    fn run(&mut self, codes: &[i64], max_tokens: usize) -> Result<Vec<f32>> {
        let input_arr = ndarray::Array3::from_shape_vec((1, max_tokens, 16), codes.to_vec())?;
        let input_tensor = ort::value::Tensor::from_array(input_arr)?;
        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let audio = outputs[0]
            .try_extract_array::<f32>()
            .context("Extract vocoder output")?;
        Ok(audio.as_slice().unwrap_or(&[]).to_vec())
    }
}
