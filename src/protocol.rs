use serde::{Deserialize, Serialize};

/// Messages sent from Rust orchestrator to workers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Request {
    /// Initialize worker with model paths
    #[serde(rename = "init")]
    Init { role: String, models_dir: String },

    /// Tokenize text and build prefix embeddings
    #[serde(rename = "tokenize_and_embed")]
    TokenizeAndEmbed {
        text: String,
        language: String,
        /// Base64-encoded ref_codec_tokens for voice cloning
        ref_codec_tokens: Option<String>,
    },

    /// Prefill talker with prefix embeddings
    #[serde(rename = "talker_prefill")]
    TalkerPrefill {
        /// Base64-encoded float32 array [N, 1024]
        prefix_embeddings: String,
    },

    /// One talker decode step
    #[serde(rename = "talker_step")]
    TalkerStep {
        /// Base64-encoded float32 array [1024] - feedback embedding
        feedback_embedding: String,
        /// Sampling parameters
        temperature: f32,
        repetition_penalty: f32,
        eos_boost: f32,
        /// Past code_0 tokens for repetition penalty
        past_tokens: Vec<i32>,
    },

    /// Predict codes 1-15 from hidden state
    #[serde(rename = "code_predict")]
    CodePredict {
        /// Base64-encoded float32 array [1024]
        hidden_state: String,
        code_0: i32,
        temperature: f32,
    },

    /// Convert codes to audio
    #[serde(rename = "vocode")]
    Vocode {
        /// Base64-encoded int64 array [N, 16]
        codes: String,
        n_tokens: usize,
    },

    /// Load voice profile for cloning
    #[serde(rename = "load_voice")]
    LoadVoice {
        codec_tokens: String,
        n_tokens: usize,
        ref_text: Option<String>,
    },

    /// Health check
    #[serde(rename = "ping")]
    Ping,
}

/// Worker responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub status: String,
    #[serde(default)]
    pub data: serde_json::Value,
    #[serde(default)]
    pub error: Option<String>,
}

// Constants matching the Python pipeline
pub const HIDDEN_SIZE: usize = 1024;
pub const CODEC_EOS_ID: i32 = 2150;
pub const SAMPLE_RATE: u32 = 24000;
pub const SAMPLES_PER_TOKEN: usize = 1920;
