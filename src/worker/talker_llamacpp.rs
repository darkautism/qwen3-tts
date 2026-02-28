//! Talker LLM via llama.cpp shared library (libloading).
//! Only compiled with `ggml-backend` feature.

use anyhow::{Context, Result};
use ndarray::Array1;
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::Path;

type WrapperBackendInit = unsafe extern "C" fn();
type WrapperBackendFree = unsafe extern "C" fn();
type WrapperLoadModel = unsafe extern "C" fn(*const c_char, c_int) -> *mut c_void;
type WrapperFreeModel = unsafe extern "C" fn(*mut c_void);
type WrapperModelNEmbd = unsafe extern "C" fn(*const c_void) -> c_int;
type WrapperCreateContext =
    unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int) -> *mut c_void;
type WrapperFreeContext = unsafe extern "C" fn(*mut c_void);
type WrapperKvClear = unsafe extern "C" fn(*mut c_void);
type WrapperDecodeEmbd =
    unsafe extern "C" fn(*mut c_void, *const c_float, c_int, c_int, c_int, *mut c_float) -> c_int;

pub struct TalkerLlamaCpp {
    _lib: libloading::Library,
    model: *mut c_void,
    ctx: *mut c_void,
    n_embd: usize,
    pos: usize,
    fn_kv_clear: WrapperKvClear,
    fn_decode_embd: WrapperDecodeEmbd,
}

unsafe impl Send for TalkerLlamaCpp {}
unsafe impl Sync for TalkerLlamaCpp {}

impl TalkerLlamaCpp {
    pub fn load(model_path: &Path, n_ctx: i32, n_threads: i32) -> Result<Self> {
        let lib_path = find_wrapper_lib()?;
        tracing::info!("Loading llama_wrapper from {}", lib_path.display());

        let lib = unsafe { libloading::Library::new(&lib_path) }
            .with_context(|| format!("Load {}", lib_path.display()))?;

        unsafe {
            let init: libloading::Symbol<WrapperBackendInit> = lib.get(b"wrapper_backend_init")?;
            init();

            let load_model: libloading::Symbol<WrapperLoadModel> =
                lib.get(b"wrapper_load_model")?;
            let c_path = CString::new(model_path.to_str().unwrap())?;
            let model = load_model(c_path.as_ptr(), 0);
            if model.is_null() {
                anyhow::bail!("Failed to load model: {}", model_path.display());
            }

            let model_n_embd: libloading::Symbol<WrapperModelNEmbd> =
                lib.get(b"wrapper_model_n_embd")?;
            let n_embd = model_n_embd(model) as usize;

            let create_ctx: libloading::Symbol<WrapperCreateContext> =
                lib.get(b"wrapper_create_context")?;
            let ctx = create_ctx(model, n_ctx, 512, n_threads, 1 /* embeddings=true */);
            if ctx.is_null() {
                anyhow::bail!("Failed to create context");
            }

            let fn_kv_clear: WrapperKvClear = *lib.get::<WrapperKvClear>(b"wrapper_kv_clear")?;
            let fn_decode_embd: WrapperDecodeEmbd =
                *lib.get::<WrapperDecodeEmbd>(b"wrapper_decode_embd")?;

            tracing::info!("TalkerLlamaCpp ready: n_embd={} (llama.cpp)", n_embd);
            Ok(Self {
                _lib: lib,
                model,
                ctx,
                n_embd,
                pos: 0,
                fn_kv_clear,
                fn_decode_embd,
            })
        }
    }

    pub fn get_hidden(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        keep_history: bool,
    ) -> Result<Array1<f32>> {
        if !keep_history {
            unsafe { (self.fn_kv_clear)(self.ctx) };
            self.pos = 0;
        }

        assert_eq!(embeddings.len(), n_tokens * self.n_embd);
        let mut out = vec![0f32; self.n_embd];

        let ret = unsafe {
            (self.fn_decode_embd)(
                self.ctx,
                embeddings.as_ptr(),
                n_tokens as c_int,
                self.n_embd as c_int,
                self.pos as c_int,
                out.as_mut_ptr(),
            )
        };
        if ret != 0 {
            anyhow::bail!("llama.cpp decode failed with code {}", ret);
        }
        self.pos += n_tokens;
        Ok(Array1::from_vec(out))
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
}

impl Drop for TalkerLlamaCpp {
    fn drop(&mut self) {
        unsafe {
            if let Ok(lib) = libloading::Library::new(find_wrapper_lib().unwrap()) {
                if let Ok(free_ctx) =
                    lib.get::<unsafe extern "C" fn(*mut c_void)>(b"wrapper_free_context")
                {
                    free_ctx(self.ctx);
                }
                if let Ok(free_model) =
                    lib.get::<unsafe extern "C" fn(*mut c_void)>(b"wrapper_free_model")
                {
                    free_model(self.model);
                }
                if let Ok(backend_free) = lib.get::<unsafe extern "C" fn()>(b"wrapper_backend_free")
                {
                    backend_free();
                }
            }
        }
    }
}

fn find_wrapper_lib() -> Result<std::path::PathBuf> {
    let candidates = [
        "/usr/lib/llama_wrapper.so",
        "/usr/local/lib/llama_wrapper.so",
        "./llama_wrapper.so",
    ];
    for p in &candidates {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    anyhow::bail!(
        "llama_wrapper.so not found. Install to /usr/lib/ or set LD_LIBRARY_PATH. \
         See README for build instructions."
    )
}
