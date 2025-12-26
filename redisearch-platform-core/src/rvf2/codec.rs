//! Codec implementations: SQ8, FP16, PQ16
//!
//! Provides encoding and decoding for various vector compression formats.

use half::f16;

use crate::rvf2::{Result, Rvf2Error, DEFAULT_BLOCK_SIZE};

pub use crate::rvf2::manifest::CodecType;

/// Codec trait for encoding/decoding vectors
pub trait Codec: Send + Sync {
    /// Encode float32 vectors to compressed format
    fn encode(&self, vectors: &[f32], dims: usize) -> Result<EncodedVectors>;

    /// Decode compressed format back to float32
    fn decode(&self, encoded: &EncodedVectors, output: &mut [f32]) -> Result<()>;

    /// Codec type identifier
    fn codec_type(&self) -> CodecType;

    /// Bytes per vector (approximate, for capacity planning)
    fn bytes_per_vector(&self, dims: usize) -> usize;
}

/// Encoded vectors with optional auxiliary data
#[derive(Debug, Clone)]
pub struct EncodedVectors {
    /// Encoded vector codes
    pub codes: Vec<u8>,

    /// Auxiliary data (scales for SQ8, codebook refs for PQ, etc.)
    pub aux: Vec<u8>,

    /// Number of vectors
    pub n_vectors: usize,

    /// Dimension
    pub dims: usize,
}

/// SQ8 (Scalar Quantization, int8) codec
pub struct Sq8Codec {
    /// Block size for scale computation
    pub block_size: usize,
}

impl Default for Sq8Codec {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }
}

impl Sq8Codec {
    /// Create with custom block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Encode a single vector to int8
    #[inline]
    pub fn encode_vector(vector: &[f32], scale: f32, output: &mut [i8]) {
        let inv_scale = if scale > 0.0 { 127.0 / scale } else { 0.0 };
        for (i, &v) in vector.iter().enumerate() {
            output[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    /// Decode a single vector from int8
    #[inline]
    pub fn decode_vector(codes: &[i8], scale: f32, output: &mut [f32]) {
        let scale_factor = scale / 127.0;
        for (i, &c) in codes.iter().enumerate() {
            output[i] = c as f32 * scale_factor;
        }
    }

    /// Compute scale for a block of vectors (max abs value)
    pub fn compute_block_scale(vectors: &[f32]) -> f32 {
        vectors
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b))
    }
}

impl Codec for Sq8Codec {
    fn encode(&self, vectors: &[f32], dims: usize) -> Result<EncodedVectors> {
        let n_vectors = vectors.len() / dims;
        let n_blocks = (n_vectors + self.block_size - 1) / self.block_size;

        let mut codes = Vec::with_capacity(n_vectors * dims);
        let mut scales_f16 = Vec::with_capacity(n_blocks * 2);

        for block_idx in 0..n_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(n_vectors);

            // Compute scale for this block
            let block_start = start * dims;
            let block_end = end * dims;
            let scale = Self::compute_block_scale(&vectors[block_start..block_end]);

            // Store scale as f16
            let scale_f16 = f16::from_f32(scale);
            scales_f16.extend_from_slice(&scale_f16.to_le_bytes());

            // Quantize vectors in block
            let inv_scale = if scale > 0.0 { 127.0 / scale } else { 0.0 };
            for i in start..end {
                let vec_start = i * dims;
                for d in 0..dims {
                    let v = vectors[vec_start + d];
                    let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    codes.push(q as u8);
                }
            }
        }

        Ok(EncodedVectors {
            codes,
            aux: scales_f16,
            n_vectors,
            dims,
        })
    }

    fn decode(&self, encoded: &EncodedVectors, output: &mut [f32]) -> Result<()> {
        if output.len() < encoded.n_vectors * encoded.dims {
            return Err(Rvf2Error::Codec("Output buffer too small".into()));
        }

        let n_blocks = (encoded.n_vectors + self.block_size - 1) / self.block_size;

        for block_idx in 0..n_blocks {
            // Read scale
            let scale_offset = block_idx * 2;
            let scale = f16::from_le_bytes([
                encoded.aux[scale_offset],
                encoded.aux[scale_offset + 1],
            ])
            .to_f32();
            let scale_factor = scale / 127.0;

            // Decode vectors in block
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(encoded.n_vectors);

            for i in start..end {
                let code_start = i * encoded.dims;
                let out_start = i * encoded.dims;
                for d in 0..encoded.dims {
                    let code = encoded.codes[code_start + d] as i8;
                    output[out_start + d] = code as f32 * scale_factor;
                }
            }
        }

        Ok(())
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Sq8
    }

    fn bytes_per_vector(&self, dims: usize) -> usize {
        // dims bytes for codes + ~2/block_size bytes for scale
        dims + 2 / self.block_size + 1
    }
}

/// FP16 codec (float16)
pub struct Fp16Codec;

impl Codec for Fp16Codec {
    fn encode(&self, vectors: &[f32], dims: usize) -> Result<EncodedVectors> {
        let n_vectors = vectors.len() / dims;
        let mut codes = Vec::with_capacity(n_vectors * dims * 2);

        for &v in vectors {
            let f16_val = f16::from_f32(v);
            codes.extend_from_slice(&f16_val.to_le_bytes());
        }

        Ok(EncodedVectors {
            codes,
            aux: Vec::new(),
            n_vectors,
            dims,
        })
    }

    fn decode(&self, encoded: &EncodedVectors, output: &mut [f32]) -> Result<()> {
        if output.len() < encoded.n_vectors * encoded.dims {
            return Err(Rvf2Error::Codec("Output buffer too small".into()));
        }

        for i in 0..encoded.n_vectors * encoded.dims {
            let offset = i * 2;
            let f16_val = f16::from_le_bytes([encoded.codes[offset], encoded.codes[offset + 1]]);
            output[i] = f16_val.to_f32();
        }

        Ok(())
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Fp16
    }

    fn bytes_per_vector(&self, dims: usize) -> usize {
        dims * 2
    }
}

/// FP32 codec (no compression, passthrough)
pub struct Fp32Codec;

impl Codec for Fp32Codec {
    fn encode(&self, vectors: &[f32], dims: usize) -> Result<EncodedVectors> {
        let n_vectors = vectors.len() / dims;
        let codes: Vec<u8> = vectors
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        Ok(EncodedVectors {
            codes,
            aux: Vec::new(),
            n_vectors,
            dims,
        })
    }

    fn decode(&self, encoded: &EncodedVectors, output: &mut [f32]) -> Result<()> {
        if output.len() < encoded.n_vectors * encoded.dims {
            return Err(Rvf2Error::Codec("Output buffer too small".into()));
        }

        for i in 0..encoded.n_vectors * encoded.dims {
            let offset = i * 4;
            output[i] = f32::from_le_bytes([
                encoded.codes[offset],
                encoded.codes[offset + 1],
                encoded.codes[offset + 2],
                encoded.codes[offset + 3],
            ]);
        }

        Ok(())
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Fp32
    }

    fn bytes_per_vector(&self, dims: usize) -> usize {
        dims * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_roundtrip() {
        let codec = Sq8Codec::default();
        let dims = 128;
        let n_vectors = 64;

        // Generate test vectors
        let vectors: Vec<f32> = (0..n_vectors * dims)
            .map(|i| (i as f32 / 100.0).sin())
            .collect();

        // Encode
        let encoded = codec.encode(&vectors, dims).unwrap();
        assert_eq!(encoded.n_vectors, n_vectors);
        assert_eq!(encoded.codes.len(), n_vectors * dims);

        // Decode
        let mut decoded = vec![0.0f32; n_vectors * dims];
        codec.decode(&encoded, &mut decoded).unwrap();

        // Check approximate equality (SQ8 is lossy)
        for i in 0..vectors.len() {
            let diff = (vectors[i] - decoded[i]).abs();
            assert!(diff < 0.02, "diff {} at index {}", diff, i);
        }
    }

    #[test]
    fn test_fp16_roundtrip() {
        let codec = Fp16Codec;
        let dims = 128;
        let n_vectors = 16;

        let vectors: Vec<f32> = (0..n_vectors * dims)
            .map(|i| (i as f32 / 100.0).sin())
            .collect();

        let encoded = codec.encode(&vectors, dims).unwrap();
        let mut decoded = vec![0.0f32; n_vectors * dims];
        codec.decode(&encoded, &mut decoded).unwrap();

        for i in 0..vectors.len() {
            let diff = (vectors[i] - decoded[i]).abs();
            assert!(diff < 0.001, "diff {} at index {}", diff, i);
        }
    }
}

