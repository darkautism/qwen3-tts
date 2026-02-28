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

/// Files needed for the predictor role
const PREDICTOR_FILES: &[&str] = &[
    "predictor/code_predictor/qwen3-tts-0.6b-q8_0.gguf",
    "predictor/code_predictor/code_predictor_weights.npz",
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

    let files = match role {
        "talker" => TALKER_FILES,
        "predictor" => PREDICTOR_FILES,
        "vocoder" => VOCODER_FILES,
        _ => anyhow::bail!("Unknown role: {}", role),
    };

    // Check if all files exist
    let missing: Vec<&&str> = files
        .iter()
        .filter(|f| {
            let local = models_dir.join(f);
            !local.exists()
        })
        .collect();

    if missing.is_empty() {
        info!("All {} model files present in {}", role, role_dir.display());
        return Ok(role_dir);
    }

    info!(
        "Downloading {} missing files for {} from {}...",
        missing.len(),
        role,
        repo
    );

    let api = Api::new().context("Initialize HuggingFace Hub API")?;
    let repo_handle = api.model(repo.to_string());

    for &file in &missing {
        info!("  ↓ {}", file);
        let cached_path = repo_handle
            .get(file)
            .with_context(|| format!("Download {}", file))?;

        // hf-hub caches to ~/.cache/huggingface; symlink/copy to our models dir
        let local_path = models_dir.join(file);
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create symlink if possible, otherwise copy
        #[cfg(unix)]
        {
            if local_path.exists() || local_path.is_symlink() {
                std::fs::remove_file(&local_path).ok();
            }
            std::os::unix::fs::symlink(&cached_path, &local_path).unwrap_or_else(|_| {
                std::fs::copy(&cached_path, &local_path).expect("Failed to copy model file");
            });
        }
        #[cfg(not(unix))]
        {
            std::fs::copy(&cached_path, &local_path)?;
        }
    }

    info!("✓ All {} model files ready", role);
    Ok(role_dir)
}
