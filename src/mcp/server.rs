use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

use crate::audio;
use crate::config::Config;
use crate::pipeline::{Pipeline, SynthesisParams};

/// MCP server using stdio JSON-RPC transport
pub struct McpServer {
    pipeline: Arc<Mutex<Pipeline>>,
    config: Config,
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

impl McpServer {
    pub fn new(pipeline: Arc<Mutex<Pipeline>>, config: Config) -> Self {
        Self { pipeline, config }
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);

        tracing::info!("MCP server started on stdio");

        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break; // EOF
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<JsonRpcRequest>(line) {
                Ok(req) => {
                    let resp = self.handle_request(req).await;
                    let resp_str = serde_json::to_string(&resp)? + "\n";
                    stdout.write_all(resp_str.as_bytes()).await?;
                    stdout.flush().await?;
                }
                Err(e) => {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: Value::Null,
                        result: None,
                        error: Some(json!({
                            "code": -32700,
                            "message": format!("Parse error: {}", e),
                        })),
                    };
                    let resp_str = serde_json::to_string(&resp)? + "\n";
                    stdout.write_all(resp_str.as_bytes()).await?;
                    stdout.flush().await?;
                }
            }
        }

        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.unwrap_or(Value::Null);

        match req.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "qwen3-tts",
                        "version": "0.1.0"
                    }
                })),
                error: None,
            },

            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "tools": [
                        {
                            "name": "text_to_speech",
                            "description": "Convert text to speech audio. Returns a WAV file path.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "Text to synthesize"
                                    },
                                    "language": {
                                        "type": "string",
                                        "description": "Language code",
                                        "enum": ["chinese", "english", "russian", "german", "french", "japanese", "korean"],
                                        "default": "chinese"
                                    },
                                    "output_path": {
                                        "type": "string",
                                        "description": "Output WAV file path",
                                        "default": "output.wav"
                                    },
                                    "voice": {
                                        "type": "string",
                                        "description": "Path to voice profile file (.json from encode-voice). Create with `qwen3-tts encode-voice -a audio.wav -r \"text\" -o voice.json`."
                                    }
                                },
                                "required": ["text"]
                            }
                        }
                    ]
                })),
                error: None,
            },

            "tools/call" => {
                let tool_name = req.params["name"].as_str().unwrap_or("");
                let args = &req.params["arguments"];

                match tool_name {
                    "text_to_speech" => self.tool_tts(id, args).await,
                    _ => JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id,
                        result: None,
                        error: Some(json!({
                            "code": -32601,
                            "message": format!("Unknown tool: {}", tool_name),
                        })),
                    },
                }
            }

            "notifications/initialized" | "initialized" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({})),
                error: None,
            },

            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(json!({
                    "code": -32601,
                    "message": format!("Method not found: {}", req.method),
                })),
            },
        }
    }

    async fn tool_tts(&self, id: Value, args: &Value) -> JsonRpcResponse {
        let text = args["text"].as_str().unwrap_or("").to_string();
        let language = args["language"]
            .as_str()
            .unwrap_or(&self.config.defaults.language)
            .to_string();
        let output_path = args["output_path"]
            .as_str()
            .unwrap_or("output.wav")
            .to_string();
        let voice = match args["voice"].as_str() {
            Some(v) => match crate::voices::resolve_voice(v) {
                Ok(p) => Some(p.to_string_lossy().to_string()),
                Err(e) => {
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id,
                        result: None,
                        error: Some(json!({
                            "code": -32602,
                            "message": format!("Voice '{}' not found: {}", v, e),
                        })),
                    };
                }
            },
            None => None,
        };

        let params = SynthesisParams {
            text,
            language,
            voice,
            max_tokens: self.config.defaults.max_tokens,
            temperature: self.config.defaults.temperature,
            cp_temperature: self.config.defaults.cp_temperature,
            repetition_penalty: self.config.defaults.repetition_penalty,
        };

        let mut pipeline = self.pipeline.lock().await;
        match pipeline.synthesize(&params).await {
            Ok(result) => {
                if let Err(e) =
                    audio::save_wav(&result.audio_samples, &output_path, result.sample_rate)
                {
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id,
                        result: Some(json!({
                            "content": [{
                                "type": "text",
                                "text": format!("Error saving WAV: {}", e)
                            }],
                            "isError": true,
                        })),
                        error: None,
                    };
                }

                let duration = result.audio_samples.len() as f32 / result.sample_rate as f32;
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: Some(json!({
                        "content": [{
                            "type": "text",
                            "text": format!(
                                "Speech generated: {} ({:.1}s, {} tokens, {:.0}ms)",
                                output_path, duration, result.n_tokens, result.generation_time_ms
                            )
                        }]
                    })),
                    error: None,
                }
            }
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Synthesis failed: {}", e)
                    }],
                    "isError": true,
                })),
                error: None,
            },
        }
    }
}
