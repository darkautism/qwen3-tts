use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::protocol::{Request, Response};

/// Client for communicating with a Python inference worker
pub struct WorkerClient {
    stream: TcpStream,
    addr: String,
}

impl WorkerClient {
    /// Connect to a worker at the given address
    pub async fn connect(host: &str, port: u16) -> Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr)
            .await
            .with_context(|| format!("Failed to connect to worker at {}", addr))?;
        stream.set_nodelay(true)?;
        tracing::info!("Connected to worker at {}", addr);
        Ok(Self { stream, addr })
    }

    /// Send a request and receive a response
    pub async fn call(&mut self, req: &Request) -> Result<Response> {
        let payload = rmp_serde::to_vec_named(req).context("Failed to serialize request")?;

        // Send: [4 bytes BE length][payload]
        let len = payload.len() as u32;
        self.stream.write_all(&len.to_be_bytes()).await?;
        self.stream.write_all(&payload).await?;
        self.stream.flush().await?;

        // Receive: [4 bytes BE length][payload]
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .with_context(|| format!("Worker {} disconnected", self.addr))?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        let mut resp_buf = vec![0u8; resp_len];
        self.stream.read_exact(&mut resp_buf).await?;

        let resp: Response =
            rmp_serde::from_slice(&resp_buf).context("Failed to deserialize worker response")?;

        if resp.status != "ok" {
            anyhow::bail!(
                "Worker error: {}",
                resp.error.as_deref().unwrap_or("unknown")
            );
        }

        Ok(resp)
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }
}

// Helper to encode binary data as base64
pub fn encode_f32(data: &[f32]) -> String {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    B64.encode(&bytes)
}

pub fn encode_i64(data: &[i64]) -> String {
    let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
    B64.encode(&bytes)
}

pub fn decode_f32(b64: &str) -> Result<Vec<f32>> {
    let bytes = B64.decode(b64)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn decode_i16(b64: &str) -> Result<Vec<i16>> {
    let bytes = B64.decode(b64)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}
