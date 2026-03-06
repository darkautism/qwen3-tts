use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use qwen3_tts_rs::api;
use qwen3_tts_rs::audio;
use qwen3_tts_rs::cli::{Cli, Commands, WorkerRole};
use qwen3_tts_rs::config::{default_config_dir, Config};
use qwen3_tts_rs::mcp;
use qwen3_tts_rs::pipeline::{Pipeline, SynthesisParams};
use qwen3_tts_rs::voices;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qwen3_tts_rs=info".into()),
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
            chunk,
        }) => cmd_speak(text, output, lang, voice, max_tokens, None, chunk).await,

        Some(Commands::Serve { port, mcp }) => cmd_serve(port, mcp).await,

        Some(Commands::Worker {
            bind,
            role,
            models,
            repo,
            cores,
            big_cores,
            quant,
        }) => cmd_worker(bind, role, models, repo, cores, big_cores, quant).await,

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
                cmd_speak(
                    text,
                    "output.wav".into(),
                    None,
                    None,
                    None,
                    None,
                    "none".into(),
                )
                .await
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
    chunk: String,
) -> Result<()> {
    let config = Config::load(config_path)?;
    let language = lang.unwrap_or_else(|| config.defaults.language.clone());
    let max_tok = max_tokens.unwrap_or(config.defaults.max_tokens);
    let chunk_mode: qwen3_tts_rs::pipeline::ChunkMode =
        chunk.parse().map_err(|e: String| anyhow::anyhow!(e))?;

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
            voice_data: None,
            max_tokens: max_tok,
            temperature: config.defaults.temperature,
            cp_temperature: config.defaults.cp_temperature,
            repetition_penalty: config.defaults.repetition_penalty,
            chunk_mode,
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
    let (config, config_path) = Config::load_with_path(None)?;
    let listen_port = port.unwrap_or(config.server.port);
    let synthesis_lock = Arc::new(Mutex::new(()));

    if with_mcp {
        let mcp_pipeline = Arc::new(Mutex::new(Pipeline::new(config.clone()).await?));
        let mcp_config = config.clone();
        let mcp_schedule_lock = synthesis_lock.clone();
        tokio::spawn(async move {
            let server = mcp::server::McpServer::new(mcp_pipeline, mcp_config, mcp_schedule_lock);
            if let Err(e) = server.run().await {
                tracing::error!("MCP server error: {}", e);
            }
        });
    }

    let state = Arc::new(api::openai::AppState {
        config: Mutex::new(config.clone()),
        config_path,
        synthesis_lock,
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
    cores: Option<String>,
    big_cores: bool,
    quant: String,
) -> Result<()> {
    // Pin CPU affinity before loading models (affects all threads)
    if let Some(ref core_spec) = cores {
        set_cpu_affinity(core_spec)?;
    } else {
        let big = detect_big_cores();
        if !big.is_empty() {
            let spec = big
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",");
            set_cpu_affinity(&spec)?;
            if big_cores {
                info!("--big-cores specified (same as default auto big-core pinning)");
            } else {
                info!("Auto big-core pinning enabled by default");
            }
        } else {
            if big_cores {
                info!("--big-cores set but no big cores detected, using all cores");
            } else {
                info!("No big cores detected, using all cores");
            }
        }
    }

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

    // Set quant preference as env var for worker and download to pick up
    std::env::set_var("QWEN3_TTS_QUANT", &quant);

    // Auto-download missing model files
    let role_dir = qwen3_tts_rs::download::ensure_models(role_str, &models_dir, Some(&repo))?;

    qwen3_tts_rs::worker::run_worker(&bind, role_str, role_dir.to_str().unwrap()).await
}

/// Parse core spec ("4,5" or "4-7" or "4,5,6-7") and set CPU affinity via sched_setaffinity
#[cfg(target_os = "linux")]
fn set_cpu_affinity(spec: &str) -> Result<()> {
    let mut cores = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start.trim().parse()?;
            let e: usize = end.trim().parse()?;
            for c in s..=e {
                cores.push(c);
            }
        } else {
            cores.push(part.parse()?);
        }
    }
    if cores.is_empty() {
        anyhow::bail!("No cores specified");
    }

    // Build CPU_SET bitmask
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in &cores {
            libc::CPU_SET(c, &mut set);
        }

        // Set affinity for ALL threads in this process (tokio spawns threads before main)
        let task_dir = std::path::Path::new("/proc/self/task");
        if task_dir.exists() {
            for entry in std::fs::read_dir(task_dir)? {
                if let Ok(entry) = entry {
                    if let Ok(tid) = entry.file_name().to_string_lossy().parse::<i32>() {
                        libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
                    }
                }
            }
        } else {
            // Fallback: set for current thread only
            let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            if ret != 0 {
                anyhow::bail!(
                    "sched_setaffinity failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }

    // Limit rayon thread pool to match pinned core count
    std::env::set_var("RAYON_NUM_THREADS", cores.len().to_string());

    info!("CPU affinity set to cores: {:?}", cores);
    Ok(())
}

/// Detect big (performance) CPU cores by reading /sys/devices/system/cpu/cpu*/cpu_capacity
/// On RK3588: A76 cores report capacity=1024, A55 cores report capacity=446
#[cfg(target_os = "linux")]
fn detect_big_cores() -> Vec<usize> {
    let mut big = Vec::new();
    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    for i in 0..16 {
        let cap_path = cpu_dir.join(format!("cpu{}", i)).join("cpu_capacity");
        if let Ok(s) = std::fs::read_to_string(&cap_path) {
            if let Ok(cap) = s.trim().parse::<u32>() {
                if cap >= 800 {
                    big.push(i);
                }
            }
        }
    }
    big
}

#[cfg(not(target_os = "linux"))]
fn set_cpu_affinity(_spec: &str) -> Result<()> {
    anyhow::bail!("CPU affinity pinning is only supported on Linux")
}

#[cfg(not(target_os = "linux"))]
fn detect_big_cores() -> Vec<usize> {
    Vec::new()
}

async fn cmd_mcp() -> Result<()> {
    let config = Config::load(None)?;
    let pipeline = Pipeline::new(config.clone()).await?;
    let pipeline = Arc::new(Mutex::new(pipeline));
    let synthesis_lock = Arc::new(Mutex::new(()));
    let server = mcp::server::McpServer::new(pipeline, config, synthesis_lock);
    server.run().await
}

async fn cmd_init(
    talker_ip: String,
    predictor_ip: String,
    vocoder_ip: Option<String>,
) -> Result<()> {
    let config_dir = default_config_dir();
    std::fs::create_dir_all(&config_dir)?;
    let config_path = config_dir.join("config.toml");

    let (vocoder_host, vocoder_port) = if let Some(ref vip) = vocoder_ip {
        if let Some(idx) = vip.rfind(':') {
            let port_str = &vip[idx + 1..];
            if let Ok(p) = port_str.parse::<u16>() {
                (vip[..idx].to_string(), p)
            } else {
                (vip.clone(), 9092u16)
            }
        } else {
            (vip.clone(), 9092u16)
        }
    } else {
        (predictor_ip.clone(), 9092u16)
    };

    let content = format!(
        r#"# Qwen3-TTS 配置檔 (自動生成)
#
# Talker:    Tokenizer + TextEmbedder + Talker LLM
# Predictor: CodePredictor (ONNX)
# Vocoder:   Vocoder (RKNN/ONNX)

[models]
dir = "~/.local/share/qwen3-tts/models"

[workers.talker]
host = "{talker_ip}"
port = 9090

[workers.predictor]
host = "{predictor_ip}"
port = 9091

[workers.vocoder]
host = "{vocoder_host}"
port = {vocoder_port}

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
    println!("  Vocoder   : {}:{}", vocoder_host, vocoder_port);
    println!();
    println!("下一步:");
    println!("  # 各機器啟動 worker");
    println!("  qwen3-tts worker -r talker");
    println!("  qwen3-tts worker -r predictor");
    println!("  qwen3-tts worker -r vocoder");
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
    let tokenizer = qwen3_tts_rs::speech_tokenizer::SpeechTokenizer::load(&model_path)?;
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
    qwen3_tts_rs::speech_tokenizer::resolve_model_path(hf_model)
}
