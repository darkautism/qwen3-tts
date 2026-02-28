use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "qwen3-tts", about = "Distributed Qwen3-TTS on RK3588")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Text to synthesize (shorthand for `speak` subcommand)
    #[arg(trailing_var_arg = true)]
    pub text: Vec<String>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Synthesize speech from text
    Speak {
        /// Text to synthesize
        text: String,

        /// Output WAV file
        #[arg(short, long, default_value = "output.wav")]
        output: String,

        /// Language
        #[arg(short, long)]
        lang: Option<String>,

        /// Voice profile file for cloning (.json from encode-voice, or .npy/.pt)
        #[arg(short, long)]
        voice: Option<String>,

        /// Max tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,
    },

    /// Start API server (OpenAI-compatible + MCP)
    Serve {
        /// Listen port
        #[arg(short, long)]
        port: Option<u16>,

        /// Also start MCP on stdio
        #[arg(long)]
        mcp: bool,
    },

    /// Run as inference worker
    Worker {
        /// Bind address
        #[arg(short, long, default_value = "0.0.0.0:9090")]
        bind: String,

        /// Worker role
        #[arg(short, long)]
        role: WorkerRole,

        /// Models directory (default: ~/.local/share/qwen3-tts/models)
        #[arg(short, long)]
        models: Option<String>,

        /// HuggingFace Hub repo (auto-download if models missing)
        #[arg(long, default_value = "kautism/qwen3-tts-rk3588")]
        repo: String,
    },

    /// Start MCP server (stdio transport)
    Mcp,

    /// Generate default config file
    Init {
        /// Predictor worker IP (code predictor machine)
        #[arg(long)]
        predictor_ip: String,

        /// Talker worker IP (local machine, usually 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        talker_ip: String,

        /// Vocoder worker IP (defaults to predictor IP if not set)
        #[arg(long)]
        vocoder_ip: Option<String>,
    },

    /// Convert models (x86 only)
    Convert {
        /// HuggingFace model name or path
        #[arg(long, default_value = "Qwen/Qwen3-TTS")]
        hf_model: String,

        /// Output directory
        #[arg(short, long, default_value = "./models")]
        output: String,

        /// Target platform
        #[arg(long, default_value = "rk3588")]
        target: String,
    },

    /// Encode reference audio to voice profile (runs natively on ARM64)
    EncodeVoice {
        /// Reference audio WAV file (24kHz mono recommended)
        #[arg(short, long)]
        audio: String,

        /// Text spoken in the reference audio (required for voice cloning quality)
        #[arg(short, long)]
        ref_text: String,

        /// Output voice profile file (.json)
        #[arg(short, long, default_value = "voice.json")]
        output: String,

        /// HuggingFace model for speech tokenizer
        #[arg(long, default_value = "kautism/qwen3-tts-rk3588")]
        hf_model: String,
    },
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum WorkerRole {
    Talker,
    Predictor,
    Vocoder,
}
