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
        let (config, _) = Self::load_with_path(path)?;
        Ok(config)
    }

    /// Load config and return the path it was loaded from
    pub fn load_with_path(path: Option<&str>) -> anyhow::Result<(Self, PathBuf)> {
        let candidates = if let Some(p) = path {
            vec![PathBuf::from(p)]
        } else {
            vec![PathBuf::from("qwen3-tts.toml"), default_config_path()]
        };

        for p in &candidates {
            if p.exists() {
                let content = std::fs::read_to_string(p)?;
                let config: Config = toml::from_str(&content)?;
                tracing::info!("Config loaded from {}", p.display());
                return Ok((config, p.clone()));
            }
        }

        anyhow::bail!(
            "No config file found. Create qwen3-tts.toml or {}",
            default_config_path().display()
        )
    }

    /// Save config to the given path
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        tracing::info!("Config saved to {}", path.display());
        Ok(())
    }
}

pub fn default_config_dir() -> PathBuf {
    dirs::config_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
        .join("qwen3-tts")
}

pub fn default_config_path() -> PathBuf {
    default_config_dir().join("config.toml")
}
