use anyhow::Result;
use std::path::{Path, PathBuf};

/// Resolve a voice to a local file path (.json, .npy, or .pt).
/// Returns an absolute path so workers with different CWDs can access it.
pub fn resolve_voice(voice: &str) -> Result<PathBuf> {
    let path = Path::new(voice);

    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("json" | "npy" | "pt") => return Ok(path.to_path_buf()),
            Some(ext) => {
                anyhow::bail!(
                    "Unsupported voice format '.{}'. Use .json, .npy or .pt",
                    ext
                )
            }
            None => anyhow::bail!("Voice file has no extension. Use .json, .npy or .pt"),
        }
    }

    anyhow::bail!(
        "Voice file '{}' not found. \
         Use `qwen3-tts encode-voice -a audio.wav -r \"text\" -o {}` to create one.",
        voice,
        voice
    );
}
