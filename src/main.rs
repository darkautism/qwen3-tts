use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use qwen3_tts::api;
use qwen3_tts::audio;
use qwen3_tts::cli::{Cli, Commands, WorkerRole};
use qwen3_tts::config::Config;
use qwen3_tts::mcp;
use qwen3_tts::pipeline::{Pipeline, SynthesisParams};
use qwen3_tts::voices;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qwen3_tts=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // If no subcommand but text provided, treat as `speak`
    match cli.command {
        Some(Commands::Speak {
            text,
            output,
            lang,
            voice,
            max_tokens,
        }) => cmd_speak(text, output, lang, voice, max_tokens, None).await,

        Some(Commands::Serve { port, mcp }) => cmd_serve(port, mcp).await,

        Some(Commands::Worker {
            bind,
            role,
            models,
            repo,
        }) => cmd_worker(bind, role, models, repo).await,

        Some(Commands::Mcp) => cmd_mcp().await,

        Some(Commands::Init {
            predictor_ip,
            talker_ip,
            vocoder_ip,
        }) => cmd_init(talker_ip, predictor_ip, vocoder_ip).await,

        Some(Commands::Convert {
            hf_model,
            output,
            target,
        }) => cmd_convert(hf_model, output, target).await,

        Some(Commands::EncodeVoice {
            audio,
            ref_text,
            output,
            hf_model,
        }) => cmd_encode_voice(audio, ref_text, output, hf_model).await,

        None => {
            if cli.text.is_empty() {
                // No text, no subcommand: print help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            } else {
                // Shorthand: qwen3-tts "hello world"
                let text = cli.text.join(" ");
                cmd_speak(text, "output.wav".into(), None, None, None, None).await
            }
        }
    }
}

async fn cmd_speak(
    text: String,
    output: String,
    lang: Option<String>,
    voice: Option<String>,
    max_tokens: Option<usize>,
    config_path: Option<&str>,
) -> Result<()> {
    let config = Config::load(config_path)?;
    let language = lang.unwrap_or_else(|| config.defaults.language.clone());
    let max_tok = max_tokens.unwrap_or(config.defaults.max_tokens);

    // Resolve voice name to file path
    let voice = match voice {
        Some(v) => {
            let resolved = voices::resolve_voice(&v)?;
            info!("Using voice: {} → {}", v, resolved.display());
            Some(resolved.to_string_lossy().to_string())
        }
        None => None,
    };

    info!("Synthesizing: \"{}\" [{}]", text, language);

    let mut pipeline = Pipeline::new(config.clone()).await?;
    let result = pipeline
        .synthesize(&SynthesisParams {
            text,
            language,
            voice,
            max_tokens: max_tok,
            temperature: config.defaults.temperature,
            cp_temperature: config.defaults.cp_temperature,
            repetition_penalty: config.defaults.repetition_penalty,
        })
        .await?;

    audio::save_wav(&result.audio_samples, &output, result.sample_rate)?;
    let duration = result.audio_samples.len() as f32 / result.sample_rate as f32;
    println!(
        "✓ Saved {} ({:.1}s audio, {} tokens, {:.0}ms)",
        output, duration, result.n_tokens, result.generation_time_ms
    );

    Ok(())
}

async fn cmd_serve(port: Option<u16>, with_mcp: bool) -> Result<()> {
    let config = Config::load(None)?;
    let listen_port = port.unwrap_or(config.server.port);

    let pipeline = Pipeline::new(config.clone()).await?;
    let pipeline = Arc::new(Mutex::new(pipeline));

    if with_mcp {
        let mcp_pipeline = pipeline.clone();
        let mcp_config = config.clone();
        tokio::spawn(async move {
            let server = mcp::server::McpServer::new(mcp_pipeline, mcp_config);
            if let Err(e) = server.run().await {
                tracing::error!("MCP server error: {}", e);
            }
        });
    }

    let state = Arc::new(api::openai::AppState {
        pipeline: pipeline.clone(),
        config: config.clone(),
    });

    let app = api::openai::router(state);

    let addr = format!("{}:{}", config.server.host, listen_port);
    info!("API server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn cmd_worker(
    bind: String,
    role: WorkerRole,
    models: Option<String>,
    repo: String,
) -> Result<()> {
    let role_str = match role {
        WorkerRole::Talker => "talker",
        WorkerRole::Predictor => "predictor",
        WorkerRole::Vocoder => "vocoder",
    };

    // Default models dir: ~/.local/share/qwen3-tts/models
    let models_dir = models.map(std::path::PathBuf::from).unwrap_or_else(|| {
        dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join("qwen3-tts")
            .join("models")
    });

    // Auto-download missing model files
    let role_dir = qwen3_tts::download::ensure_models(role_str, &models_dir, Some(&repo))?;

    qwen3_tts::worker::run_worker(&bind, role_str, role_dir.to_str().unwrap()).await
}

async fn cmd_mcp() -> Result<()> {
    let config = Config::load(None)?;
    let pipeline = Pipeline::new(config.clone()).await?;
    let pipeline = Arc::new(Mutex::new(pipeline));
    let server = mcp::server::McpServer::new(pipeline, config);
    server.run().await
}

async fn cmd_init(
    talker_ip: String,
    predictor_ip: String,
    vocoder_ip: Option<String>,
) -> Result<()> {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("qwen3-tts");
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("config.toml");

    let vocoder_section = if let Some(ref vip) = vocoder_ip {
        let (host, port) = if let Some(idx) = vip.rfind(':') {
            let port_str = &vip[idx + 1..];
            if let Ok(p) = port_str.parse::<u16>() {
                (&vip[..idx], p)
            } else {
                (vip.as_str(), 9092u16)
            }
        } else {
            (vip.as_str(), 9092u16)
        };
        format!(
            r#"
[workers.vocoder]
host = "{host}"
port = {port}
"#
        )
    } else {
        String::new()
    };

    let content = format!(
        r#"# Qwen3-TTS 配置檔 (自動生成)
#
# Talker:    Tokenizer + TextEmbedder + Talker LLM
# Predictor: CodePredictor (ONNX)
# Vocoder:   Vocoder (RKNN/ONNX) — 可選獨立部署，不設則使用 predictor 端點

[models]
dir = "~/.local/share/qwen3-tts/models"

[workers.talker]
host = "{talker_ip}"
port = 9090

[workers.predictor]
host = "{predictor_ip}"
port = 9091
{vocoder_section}
[defaults]
language = "chinese"
max_tokens = 200
temperature = 0.8
cp_temperature = 0.1
repetition_penalty = 1.2

[server]
host = "0.0.0.0"
port = 8080
"#
    );

    std::fs::write(&config_path, &content)?;
    println!("✓ 配置檔已寫入: {}", config_path.display());
    println!();
    println!("  Talker    : {}:9090", talker_ip);
    println!("  Predictor : {}:9091", predictor_ip);
    if let Some(ref vip) = vocoder_ip {
        println!("  Vocoder   : {}:9092", vip);
    } else {
        println!("  Vocoder   : (與 predictor 同端點)");
    }
    println!();
    println!("下一步:");
    println!("  # 各機器啟動 worker");
    println!("  qwen3-tts worker -r talker");
    println!("  qwen3-tts worker -r predictor");
    if vocoder_ip.is_some() {
        println!("  qwen3-tts worker -r vocoder");
    }
    println!();
    println!("  # 合成語音");
    println!("  qwen3-tts speak \"你好世界\"");

    Ok(())
}

async fn cmd_convert(hf_model: String, output: String, target: String) -> Result<()> {
    info!("Converting {} for {} → {}", hf_model, target, output);

    let script_dir = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("../scripts");
    let script_paths = vec![
        script_dir.join("convert_all.sh"),
        std::path::PathBuf::from("scripts/convert_all.sh"),
    ];

    let script = script_paths
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("convert_all.sh not found"))?;

    let status = tokio::process::Command::new("bash")
        .arg(script)
        .arg("--hf-model")
        .arg(&hf_model)
        .arg("--output")
        .arg(&output)
        .arg("--target")
        .arg(&target)
        .status()
        .await?;

    if !status.success() {
        anyhow::bail!("Conversion failed with {}", status);
    }

    println!("✓ Models converted to {}", output);
    Ok(())
}

async fn cmd_encode_voice(
    audio: String,
    ref_text: String,
    output: String,
    hf_model: String,
) -> Result<()> {
    info!("Encoding voice from {}", audio);

    let model_path = resolve_speech_tokenizer_path(&hf_model)?;
    let tokenizer = qwen3_tts::speech_tokenizer::SpeechTokenizer::load(&model_path)?;
    let codes = tokenizer.encode_wav(&audio)?;
    let n_tokens = codes.len();
    let n_codebooks = if n_tokens > 0 { codes[0].len() } else { 0 };

    if let Some(parent) = std::path::Path::new(&output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let profile = serde_json::json!({
        "ref_text": ref_text,
        "codec_tokens": codes,
    });
    std::fs::write(&output, serde_json::to_string(&profile)?)?;

    let truncated: String = ref_text.chars().take(40).collect();
    println!(
        "✓ {} [{}, {}] ref_text={}",
        output, n_tokens, n_codebooks, truncated
    );
    Ok(())
}

/// Resolve speech tokenizer model.safetensors path, downloading from HF if needed.
fn resolve_speech_tokenizer_path(hf_model: &str) -> Result<String> {
    // Try local path first
    let local = format!("{}/speech_tokenizer/model.safetensors", hf_model);
    if std::path::Path::new(&local).exists() {
        return Ok(local);
    }

    // Download from HuggingFace
    info!("Downloading speech tokenizer from {}", hf_model);
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(hf_model.to_string());
    let path = repo.get("speech_tokenizer/model.safetensors")?;
    Ok(path.to_string_lossy().to_string())
}
