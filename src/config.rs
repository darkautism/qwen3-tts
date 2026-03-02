use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub models: ModelsConfig,
    pub workers: WorkersConfig,
    #[serde(default)]
    pub defaults: DefaultsConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkersConfig {
    pub talker: WorkerEndpoint,
    pub predictor: WorkerEndpoint,
    #[serde(default)]
    pub vocoder: Option<WorkerEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEndpoint {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultsConfig {
    pub language: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub cp_temperature: f32,
    pub repetition_penalty: f32,
    /// EOS boost starts at this fraction of expected tokens (default: 0.6)
    #[serde(default = "default_eos_start_ratio")]
    pub eos_start_ratio: f32,
    /// EOS boost reaches max at this fraction (default: 1.2)
    #[serde(default = "default_eos_max_ratio")]
    pub eos_max_ratio: f32,
    /// Force EOS at this fraction (default: 1.5)
    #[serde(default = "default_eos_force_ratio")]
    pub eos_force_ratio: f32,
    /// Maximum EOS logit boost (default: 25.0)
    #[serde(default = "default_eos_max_boost")]
    pub eos_max_boost: f32,
}

fn default_eos_start_ratio() -> f32 {
    0.6
}
fn default_eos_max_ratio() -> f32 {
    1.2
}
fn default_eos_force_ratio() -> f32 {
    1.5
}
fn default_eos_max_boost() -> f32 {
    25.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for DefaultsConfig {
    fn default() -> Self {
        Self {
            language: "chinese".into(),
            max_tokens: 200,
            temperature: 0.8,
            cp_temperature: 0.1,
            repetition_penalty: 1.2,
            eos_start_ratio: default_eos_start_ratio(),
            eos_max_ratio: default_eos_max_ratio(),
            eos_force_ratio: default_eos_force_ratio(),
            eos_max_boost: default_eos_max_boost(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
        }
    }
}

impl Config {
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let candidates = if let Some(p) = path {
            vec![PathBuf::from(p)]
        } else {
            vec![
                PathBuf::from("qwen3-tts.toml"),
                dirs_config().join("config.toml"),
            ]
        };

        for p in &candidates {
            if p.exists() {
                let content = std::fs::read_to_string(p)?;
                let config: Config = toml::from_str(&content)?;
                tracing::info!("Config loaded from {}", p.display());
                return Ok(config);
            }
        }

        anyhow::bail!(
            "No config file found. Create qwen3-tts.toml or ~/.config/qwen3-tts/config.toml"
        )
    }
}

fn dirs_config() -> PathBuf {
    dirs_home().join(".config").join("qwen3-tts")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
}
