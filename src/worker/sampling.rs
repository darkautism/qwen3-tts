use ndarray::Array1;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Top-k + temperature + top-p sampling from logits
pub fn sample_token(
    logits: &Array1<f32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    past_tokens: Option<&[i32]>,
    repetition_penalty: f32,
    eos_boost: f32,
) -> i32 {
    let mut logits = logits.to_owned();
    let codec_eos_id = 2150usize;

    // Suppress special tokens (2048..2150 and 2151+)
    for i in 2048..codec_eos_id {
        logits[i] = -1e10;
    }
    if codec_eos_id + 1 < logits.len() {
        for i in (codec_eos_id + 1)..logits.len() {
            logits[i] = -1e10;
        }
    }

    // EOS boost
    if eos_boost > 0.0 {
        logits[codec_eos_id] += eos_boost;
    }

    // Repetition penalty
    if let Some(past) = past_tokens {
        if repetition_penalty != 1.0 {
            let window: std::collections::HashSet<i32> =
                past.iter().rev().take(30).copied().collect();
            for &t in &window {
                let idx = t as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= repetition_penalty;
                    } else {
                        logits[idx] *= repetition_penalty;
                    }
                }
            }
        }
    }

    // Top-k: get indices of top k logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(top_k);

    // Temperature scaling + softmax
    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let mut probs: Vec<f64> = indexed
        .iter()
        .map(|&(_, v)| ((v - max_logit) / temp) as f64)
        .map(|v| v.exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-p (nucleus) filtering
    if top_p < 1.0 {
        let mut sorted_idx: Vec<usize> = (0..probs.len()).collect();
        sorted_idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumsum = 0.0;
        let mut cutoff = sorted_idx.len();
        for (i, &idx) in sorted_idx.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= top_p as f64 {
                cutoff = i + 1;
                break;
            }
        }

        let keep: Vec<usize> = sorted_idx[..cutoff].to_vec();
        let mut filtered_probs = vec![0.0f64; probs.len()];
        for &idx in &keep {
            filtered_probs[idx] = probs[idx];
        }
        let sum: f64 = filtered_probs.iter().sum();
        for p in &mut filtered_probs {
            *p /= sum;
        }
        probs = filtered_probs;
    }

    // Sample
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&probs).unwrap();
    let chosen = dist.sample(&mut rng);
    indexed[chosen].0 as i32
}

/// Simple top-k + temperature sampling (for code predictor)
pub fn sample_simple(logits: &[f32], temperature: f32, top_k: usize) -> i32 {
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(top_k);

    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let probs: Vec<f64> = indexed
        .iter()
        .map(|&(_, v)| (((v - max_logit) / temp) as f64).exp())
        .collect();
    let sum: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / sum).collect();

    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&probs).unwrap();
    indexed[dist.sample(&mut rng)].0 as i32
}
