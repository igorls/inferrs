//! Audio preprocessing: log-mel spectrogram computation.
//!
//! Implements the `Gemma4AudioFeatureExtractor` algorithm:
//!  - 16 kHz input (mono f32)
//!  - 20 ms frames (320 samples), 10 ms hop (160 samples)
//!  - 512-point RFFT → 257 magnitude bins
//!  - 128 HTK-scale mel filters (0–8 kHz)
//!  - log(mel + 1e-3)
//!  - Semicausal padding: prepend frame_length/2 = 160 zeros
//!
//! Output shape: [T, 128] as a row-major `Vec<f32>` plus the frame count T.

use anyhow::{bail, Result};
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Mel spectrogram parameters.
pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LEN: usize = 320;
pub const HOP_LEN: usize = 160;
pub const FFT_LEN: usize = 512;
pub const N_FFT_BINS: usize = FFT_LEN / 2 + 1; // 257
pub const N_MEL: usize = 128;
pub const MEL_FLOOR: f32 = 1e-3;
pub const MIN_FREQ: f32 = 0.0;
pub const MAX_FREQ: f32 = 8000.0;

// ---------------------------------------------------------------------------
// Mel filter bank (computed once)
// ---------------------------------------------------------------------------

/// Precomputed HTK mel filterbank matrix: shape [N_FFT_BINS, N_MEL].
///
/// Element `[f, m]` is the weight applied to FFT bin `f` when computing mel
/// band `m`.  Matches the `transformers.audio_utils.mel_filter_bank` output
/// with `norm=None, mel_scale="htk"`.
pub fn build_mel_filterbank() -> Vec<f32> {
    let n_bins = N_FFT_BINS;
    let n_mel = N_MEL;

    // Convert Hz to HTK mel.
    let hz_to_mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(MIN_FREQ);
    let mel_max = hz_to_mel(MAX_FREQ);

    // N_MEL + 2 linearly-spaced mel points (includes two boundary points).
    let mel_points: Vec<f32> = (0..=n_mel + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mel + 1) as f32)
        .collect();

    // Convert mel points back to Hz, then to FFT bin indices.
    let bin_freqs: Vec<f32> = (0..n_bins)
        .map(|k| k as f32 * SAMPLE_RATE as f32 / FFT_LEN as f32)
        .collect();

    let mel_hz: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Build filterbank: [n_bins, n_mel] in row-major order.
    let mut fb = vec![0.0f32; n_bins * n_mel];

    for m in 0..n_mel {
        let lower = mel_hz[m];
        let center = mel_hz[m + 1];
        let upper = mel_hz[m + 2];

        for (f, &freq) in bin_freqs.iter().enumerate() {
            let w = if freq >= lower && freq <= center {
                if center > lower {
                    (freq - lower) / (center - lower)
                } else {
                    0.0
                }
            } else if freq > center && freq <= upper {
                if upper > center {
                    (upper - freq) / (upper - center)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            fb[f * n_mel + m] = w;
        }
    }
    fb
}

// ---------------------------------------------------------------------------
// Periodic Hann window
// ---------------------------------------------------------------------------

fn hann_window(n: usize) -> Vec<f32> {
    // Periodic (DFT-even) Hann: w[k] = 0.5 - 0.5 * cos(2π k / n)
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * PI * k as f32 / n as f32).cos())
        .collect()
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Compute a log-mel spectrogram from mono 16 kHz PCM samples.
///
/// Returns `(data, num_frames)` where `data` is row-major `[num_frames, N_MEL]`.
/// Panics or errors if `samples` is empty.
pub fn compute_log_mel(samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    if samples.is_empty() {
        bail!("audio input is empty");
    }

    let fb = build_mel_filterbank();
    let window = hann_window(FRAME_LEN);

    // Semicausal padding: prepend HOP_LEN zeros so the first frame is centred
    // at t=0, matching sl.STFT(time_padding='semicausal').
    let pad = HOP_LEN; // 160
    let total_len = samples.len() + pad;

    // frame_size_for_unfold = FRAME_LEN + 1 = 321
    let frame_size_for_unfold = FRAME_LEN + 1;
    let num_frames = if total_len >= frame_size_for_unfold {
        (total_len - frame_size_for_unfold) / HOP_LEN + 1
    } else {
        0
    };

    if num_frames == 0 {
        bail!("audio too short to produce any frames");
    }

    // Set up FFT planner for real-valued FFTs (we use complex FFT + take first
    // FFT_LEN/2+1 bins).
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_LEN);

    let mut out = vec![0.0f32; num_frames * N_MEL];

    let padded_get = |i: usize| -> f32 {
        if i < pad {
            0.0
        } else {
            samples[i - pad]
        }
    };

    for t in 0..num_frames {
        let start = t * HOP_LEN;

        // Extract frame of length FRAME_LEN (= frame_size_for_unfold - 1 = 320).
        // The Python code uses frames_to_process[..., :-1] after unfolding with
        // size=321 — i.e., it discards the last sample of each 321-sample window.
        let mut frame: Vec<Complex32> = (0..FRAME_LEN)
            .map(|i| {
                let sample = padded_get(start + i);
                Complex32::new(sample * window[i], 0.0)
            })
            .collect();

        // Zero-pad from FRAME_LEN to FFT_LEN.
        frame.resize(FFT_LEN, Complex32::new(0.0, 0.0));

        // In-place FFT.
        fft.process(&mut frame);

        // Magnitude of first N_FFT_BINS = 257 bins.
        // Then multiply by mel filterbank to get mel energies.
        // out[t, m] = sum_f |X[f]| * fb[f, m]
        let row = &mut out[t * N_MEL..(t + 1) * N_MEL];

        for f in 0..N_FFT_BINS {
            let mag = frame[f].norm(); // |complex|
                                       // Dot into mel bins.
            let fb_row = &fb[f * N_MEL..(f + 1) * N_MEL];
            for (m, w) in fb_row.iter().enumerate() {
                row[m] += mag * w;
            }
        }

        // log(mel + floor)
        for v in row.iter_mut() {
            *v = (*v + MEL_FLOOR).ln();
        }
    }

    Ok((out, num_frames))
}

/// Decode audio from bytes (WAV file or raw interleaved f32 PCM).
///
/// `format` should be `"wav"` or `"pcm_f32"`.
/// WAV files are decoded via a minimal parser (supports 16-bit PCM and 32-bit float WAV).
/// For `pcm_f32`, the bytes are interpreted directly as little-endian f32 samples.
pub fn decode_audio(data: &[u8], format: &str) -> Result<Vec<f32>> {
    match format {
        "wav" => decode_wav(data),
        "pcm_f32" => {
            if !data.len().is_multiple_of(4) {
                bail!("pcm_f32 data length is not a multiple of 4");
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        other => bail!("unsupported audio format: {other}; expected 'wav' or 'pcm_f32'"),
    }
}

/// Minimal WAV decoder supporting 16-bit PCM and 32-bit IEEE float, mono or
/// stereo (downmixed to mono by averaging).
fn decode_wav(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < 44 {
        bail!("WAV too short");
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }

    // Walk chunks.
    let mut pos = 12usize;
    let mut fmt_channels = 1u16;
    let mut fmt_sample_rate = 16000u32;
    let mut fmt_audio_format = 1u16; // 1=PCM, 3=IEEE float
    let mut fmt_bits_per_sample = 16u16;
    let mut pcm_data: &[u8] = &[];

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let chunk_data = if pos + 8 + chunk_size <= data.len() {
            &data[pos + 8..pos + 8 + chunk_size]
        } else {
            &data[pos + 8..]
        };

        if chunk_id == b"fmt " && chunk_data.len() >= 16 {
            fmt_audio_format = u16::from_le_bytes(chunk_data[0..2].try_into().unwrap());
            fmt_channels = u16::from_le_bytes(chunk_data[2..4].try_into().unwrap());
            fmt_sample_rate = u32::from_le_bytes(chunk_data[4..8].try_into().unwrap());
            fmt_bits_per_sample = u16::from_le_bytes(chunk_data[14..16].try_into().unwrap());
        } else if chunk_id == b"data" {
            pcm_data = chunk_data;
        }

        pos = match 8usize
            .checked_add(chunk_size)
            .and_then(|n| pos.checked_add(n))
        {
            Some(p) => p,
            None => break,
        };
    }

    if fmt_sample_rate != SAMPLE_RATE {
        // We just warn and proceed; the encoder will run at whatever rate the
        // data actually is — callers should resample to 16 kHz beforehand.
        tracing::warn!(
            "WAV sample rate is {}, expected {}. Resample to 16 kHz for best results.",
            fmt_sample_rate,
            SAMPLE_RATE
        );
    }

    let ch = fmt_channels as usize;

    let samples: Vec<f32> = match (fmt_audio_format, fmt_bits_per_sample) {
        (1, 16) => pcm_data
            .chunks_exact(2 * ch)
            .map(|frame| {
                let sum: f32 = (0..ch)
                    .map(|c| {
                        let s = i16::from_le_bytes([frame[c * 2], frame[c * 2 + 1]]);
                        s as f32 / 32768.0
                    })
                    .sum();
                sum / ch as f32
            })
            .collect(),
        (3, 32) => pcm_data
            .chunks_exact(4 * ch)
            .map(|frame| {
                let sum: f32 = (0..ch)
                    .map(|c| f32::from_le_bytes(frame[c * 4..c * 4 + 4].try_into().unwrap()))
                    .sum();
                sum / ch as f32
            })
            .collect(),
        (fmt, bits) => {
            bail!("unsupported WAV format: audio_format={fmt}, bits_per_sample={bits}; only 16-bit PCM and 32-bit float are supported");
        }
    };

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filterbank_shape() {
        let fb = build_mel_filterbank();
        assert_eq!(fb.len(), N_FFT_BINS * N_MEL);
    }

    #[test]
    fn mel_filterbank_non_negative() {
        let fb = build_mel_filterbank();
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn compute_log_mel_short_audio() {
        // 2 seconds of silence at 16kHz → should produce ~199 frames.
        let samples = vec![0.0f32; 32000];
        let (mel, n_frames) = compute_log_mel(&samples).unwrap();
        assert_eq!(n_frames, 199);
        assert_eq!(mel.len(), 199 * N_MEL);
        // Silence → all mel bins ≈ log(0 + 1e-3) = -6.908
        let expected = MEL_FLOOR.ln();
        for &v in &mel {
            assert!((v - expected).abs() < 1e-4, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn hann_window_shape() {
        let w = hann_window(FRAME_LEN);
        assert_eq!(w.len(), FRAME_LEN);
        assert!((w[0]).abs() < 1e-6); // window starts at 0
    }
}
