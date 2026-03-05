use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};
use tracing::info;

const DEFAULT_REPO: &str = "kautism/qwen3-tts-rk3588";

/// Files needed for the talker role
const TALKER_FILES: &[&str] = &[
    "talker/talker-q8_0.gguf",
    "talker/tokenizer.json",
    "talker/embeddings/text_embedding.npy",
    "talker/embeddings/codec_embedding.npy",
    "talker/embeddings/codec_head.npy",
    "talker/embeddings/text_projection_linear_fc1_weight.npy",
    "talker/embeddings/text_projection_linear_fc1_bias.npy",
    "talker/embeddings/text_projection_linear_fc2_weight.npy",
    "talker/embeddings/text_projection_linear_fc2_bias.npy",
    "talker/embeddings/tts_pad_embed.npy",
];

const PREDICTOR_Q8_FILE: &str = "predictor/code_predictor/code-predictor-q8_0.gguf";
const PREDICTOR_Q4_FILE: &str = "predictor/code_predictor/code-predictor-q4_0.gguf";
const PREDICTOR_EMBED_FILES: &[&str] = &[
    "predictor/embeddings/codec_embedding.npy",
    "predictor/embeddings/tts_pad_embed.npy",
];

/// Files needed for the vocoder role
#[cfg(feature = "rknn-vocoder")]
const VOCODER_FILES: &[&str] = &["vocoder/vocoder.rknn"];

#[cfg(not(feature = "rknn-vocoder"))]
const VOCODER_FILES: &[&str] = &["vocoder/vocoder.onnx"];

/// Download models for a specific role from HuggingFace Hub.
/// Returns the path to the role's model directory.
pub fn ensure_models(role: &str, models_dir: &Path, repo_id: Option<&str>) -> Result<PathBuf> {
    let repo = repo_id.unwrap_or(DEFAULT_REPO);
    let role_dir = models_dir.join(role);

    let files: Vec<&str> = match role {
        "talker" => TALKER_FILES.to_vec(),
        "predictor" => predictor_files_for_quant(),
        "vocoder" => VOCODER_FILES.to_vec(),
        _ => anyhow::bail!("Unknown role: {}", role),
    };

    let local_complete = files.iter().all(|f| models_dir.join(f).exists());
    info!("Resolving {} model files from HF Hub cache ({})...", role, repo);

    let api = Api::new().context("Initialize HuggingFace Hub API")?;
    let repo_handle = api.model(repo.to_string());
    let mut cache_root: Option<PathBuf> = None;

    for &file in &files {
        let cached_path = match repo_handle.get(file) {
            Ok(p) => p,
            Err(_e) if local_complete => {
                info!(
                    "HF resolve failed for {}, using local role dir: {}",
                    file,
                    role_dir.display()
                );
                return Ok(role_dir);
            }
            Err(e) => return Err(e).with_context(|| format!("Download {}", file)),
        };

        if cache_root.is_none() {
            cache_root = infer_repo_root(&cached_path, file);
        }
    }

    let hub_role_dir = cache_root
        .map(|root| root.join(role))
        .context("Failed to resolve HuggingFace cache root")?;
    info!(
        "✓ All {} model files ready (using Hub cache): {}",
        role,
        hub_role_dir.display()
    );
    Ok(hub_role_dir)
}

fn infer_repo_root(cached_path: &Path, repo_file: &str) -> Option<PathBuf> {
    let mut root = cached_path.to_path_buf();
    let depth = Path::new(repo_file).components().count();
    for _ in 0..depth {
        root = root.parent()?.to_path_buf();
    }
    Some(root)
}

fn predictor_files_for_quant() -> Vec<&'static str> {
    let quant = std::env::var("QWEN3_TTS_QUANT").unwrap_or_else(|_| "q8".to_string());
    let gguf = if quant.eq_ignore_ascii_case("q4") {
        PREDICTOR_Q4_FILE
    } else {
        PREDICTOR_Q8_FILE
    };
    let mut files = vec![gguf];
    files.extend_from_slice(PREDICTOR_EMBED_FILES);
    files
}
