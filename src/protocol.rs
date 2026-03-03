use serde::{Deserialize, Serialize};

/// Reference codec token source for tokenization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefCodecTokens {
    /// Use tokens already loaded by Request::LoadVoice.
    Loaded,
    /// Inline raw int64 little-endian bytes (shape [N, 16]).
    Inline(Vec<u8>),
}

/// Messages sent from Rust orchestrator to workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Initialize worker with model paths
    Init { role: String, models_dir: String },

    /// Tokenize text and build prefix embeddings
    TokenizeAndEmbed {
        text: String,
        language: String,
        /// Voice cloning reference source
        ref_codec_tokens: Option<RefCodecTokens>,
    },

    /// Prefill talker with prefix embeddings
    TalkerPrefill {
        /// Raw float32 little-endian bytes [N, 1024]
        prefix_embeddings: Vec<u8>,
    },

    /// One talker decode step
    TalkerStep {
        /// Raw float32 little-endian bytes [1024] - feedback embedding.
        /// None means first decode step.
        feedback_embedding: Option<Vec<u8>>,
        /// Sampling parameters
        temperature: f32,
        repetition_penalty: f32,
        eos_boost: f32,
    },

    /// Predict codes 1-15 from hidden state
    CodePredict {
        /// Raw float32 little-endian bytes [1024]
        hidden_state: Vec<u8>,
        code_0: i32,
        temperature: f32,
    },

    /// Convert codes to audio
    Vocode {
        /// Raw int64 little-endian bytes [N, 16]
        codes: Vec<u8>,
        n_tokens: usize,
    },

    /// Load voice profile for cloning
    LoadVoice {
        /// Raw int64 little-endian bytes [N, 16]
        codec_tokens: Vec<u8>,
        n_tokens: usize,
        ref_text: Option<String>,
    },

    /// Health check
    Ping,
}

/// Typed response data — avoids serde_json::Value overhead in hot path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseData {
    /// TokenizeAndEmbed response
    TokenizeAndEmbed {
        /// Raw float32 little-endian bytes [N, 1024]
        prefix_embeddings: Vec<u8>,
        n_prefix: usize,
        expected_output_tokens: usize,
    },
    /// TalkerPrefill response
    TalkerPrefill { prefilled: usize },
    /// TalkerStep response (hot path)
    TalkerStep {
        /// Raw float32 little-endian bytes [1024]
        hidden_state: Vec<u8>,
        code_0: i32,
        is_eos: bool,
    },
    /// CodePredict response (hot path)
    CodePredict {
        codes: Vec<i64>,
        /// Raw float32 little-endian bytes [1024]
        feedback_embedding: Vec<u8>,
    },
    /// Vocode response
    Vocode {
        /// Raw i16 little-endian PCM bytes
        audio: Vec<u8>,
        n_samples: usize,
    },
    /// LoadVoice response
    LoadVoice,
    /// Ping response
    Ping,
    /// Generic ack
    Init,
}

/// Worker responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub status: String,
    pub data: ResponseData,
    #[serde(default)]
    pub error: Option<String>,
}

// Constants matching the Python pipeline
pub const HIDDEN_SIZE: usize = 1024;
pub const CODEC_EOS_ID: i32 = 2150;
pub const SAMPLE_RATE: u32 = 24000;
pub const SAMPLES_PER_TOKEN: usize = 1920;
