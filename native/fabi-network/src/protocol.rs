use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use blake3::Hash;
use iroh::endpoint::{RecvStream, SendStream};

const MAGIC: [u8; 8] = *b"FABINET1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 60;
const IO_CHUNK_LEN: usize = 1024 * 1024;
const MAX_RPC_METHOD_LEN: usize = 256;
pub const DEFAULT_MAX_PAYLOAD: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageKind {
    Upload = 1,
    Acknowledgement = 2,
    Ping = 3,
    Pong = 4,
    RpcRequest = 10,
    RpcResponse = 11,
    RpcError = 12,
    RpcStreamRequest = 13,
    RpcStreamChunk = 14,
    RpcStreamEnd = 15,
    RpcStreamError = 16,
}

impl TryFrom<u8> for MessageKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Upload),
            2 => Ok(Self::Acknowledgement),
            3 => Ok(Self::Ping),
            4 => Ok(Self::Pong),
            10 => Ok(Self::RpcRequest),
            11 => Ok(Self::RpcResponse),
            12 => Ok(Self::RpcError),
            13 => Ok(Self::RpcStreamRequest),
            14 => Ok(Self::RpcStreamChunk),
            15 => Ok(Self::RpcStreamEnd),
            16 => Ok(Self::RpcStreamError),
            _ => bail!("unknown Fabi network message kind {value}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub kind: MessageKind,
    pub request_id: u64,
    pub payload_len: u64,
    pub digest: [u8; 32],
}

impl Header {
    #[must_use]
    pub fn new(kind: MessageKind, request_id: u64, payload_len: u64, digest: Hash) -> Self {
        Self {
            kind,
            request_id,
            payload_len,
            digest: *digest.as_bytes(),
        }
    }

    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut encoded = [0_u8; HEADER_LEN];
        encoded[0..8].copy_from_slice(&MAGIC);
        encoded[8..10].copy_from_slice(&VERSION.to_be_bytes());
        encoded[10] = self.kind as u8;
        encoded[11] = 0;
        encoded[12..20].copy_from_slice(&self.request_id.to_be_bytes());
        encoded[20..28].copy_from_slice(&self.payload_len.to_be_bytes());
        encoded[28..60].copy_from_slice(&self.digest);
        encoded
    }

    fn decode(encoded: &[u8; HEADER_LEN]) -> Result<Self> {
        ensure!(encoded[0..8] == MAGIC, "invalid Fabi network frame magic");
        let version = u16::from_be_bytes(encoded[8..10].try_into().expect("fixed slice"));
        ensure!(
            version == VERSION,
            "unsupported Fabi network protocol version {version}"
        );
        ensure!(encoded[11] == 0, "unsupported Fabi network frame flags");
        let request_id = u64::from_be_bytes(encoded[12..20].try_into().expect("fixed slice"));
        let payload_len = u64::from_be_bytes(encoded[20..28].try_into().expect("fixed slice"));
        let digest = encoded[28..60].try_into().expect("fixed slice");
        Ok(Self {
            kind: MessageKind::try_from(encoded[10])?,
            request_id,
            payload_len,
            digest,
        })
    }
}

/// Write a complete protocol header.
///
/// # Errors
///
/// Returns an error if the QUIC stream fails while writing.
pub async fn write_header(send: &mut SendStream, header: &Header) -> Result<()> {
    send.write_all(&header.encode())
        .await
        .context("failed to write frame header")
}

/// Read and validate a protocol header before allocating for its payload.
///
/// # Errors
///
/// Returns an error for transport failures, invalid versions/flags/kinds or a
/// payload larger than `max_payload`.
pub async fn read_header(recv: &mut RecvStream, max_payload: u64) -> Result<Header> {
    let mut encoded = [0_u8; HEADER_LEN];
    recv.read_exact(&mut encoded)
        .await
        .context("failed to read frame header")?;
    let header = Header::decode(&encoded)?;
    ensure!(
        header.payload_len <= max_payload,
        "frame payload {} exceeds configured limit {max_payload}",
        header.payload_len
    );
    Ok(header)
}

#[must_use]
pub fn deterministic_chunk(request_id: u64) -> Vec<u8> {
    let mut chunk = vec![0_u8; IO_CHUNK_LEN];
    let seed = request_id.to_le_bytes();
    for (index, byte) in chunk.iter_mut().enumerate() {
        let low_byte = u8::try_from(index & 0xff).unwrap_or(0);
        *byte = seed[index % seed.len()] ^ low_byte.wrapping_mul(31);
    }
    chunk
}

#[must_use]
pub fn digest_for_payload(request_id: u64, payload_len: u64) -> Hash {
    let chunk = deterministic_chunk(request_id);
    let mut hasher = blake3::Hasher::new();
    let mut remaining = payload_len;
    while remaining > 0 {
        let length = bounded_chunk_len(remaining, chunk.len());
        hasher.update(&chunk[..length]);
        remaining -= length as u64;
    }
    hasher.finalize()
}

/// Write the deterministic benchmark payload without allocating its full size.
///
/// # Errors
///
/// Returns an error if the QUIC stream fails while writing.
pub async fn send_payload(send: &mut SendStream, request_id: u64, payload_len: u64) -> Result<()> {
    let chunk = deterministic_chunk(request_id);
    let mut remaining = payload_len;
    while remaining > 0 {
        let length = bounded_chunk_len(remaining, chunk.len());
        send.write_all(&chunk[..length])
            .await
            .context("failed to write frame payload")?;
        remaining -= length as u64;
    }
    Ok(())
}

/// Consume exactly the declared payload and validate its BLAKE3 digest.
///
/// # Errors
///
/// Returns an error if the stream ends early or the digest does not match.
pub async fn receive_and_verify_payload(
    recv: &mut RecvStream,
    header: &Header,
) -> Result<Duration> {
    let started = std::time::Instant::now();
    let mut hasher = blake3::Hasher::new();
    let mut remaining = header.payload_len;
    let mut buffer = vec![0_u8; IO_CHUNK_LEN];
    while remaining > 0 {
        let length = bounded_chunk_len(remaining, buffer.len());
        recv.read_exact(&mut buffer[..length])
            .await
            .context("frame payload ended before its declared length")?;
        hasher.update(&buffer[..length]);
        remaining -= length as u64;
    }
    let actual = hasher.finalize();
    ensure!(
        actual.as_bytes() == &header.digest,
        "frame payload digest mismatch for request {}",
        header.request_id
    );
    Ok(started.elapsed())
}

/// Read a complete bounded payload and validate its digest.
///
/// # Errors
///
/// Returns an error if allocation is impossible, the stream ends early or the
/// digest does not match.
pub async fn read_and_verify_payload(recv: &mut RecvStream, header: &Header) -> Result<Vec<u8>> {
    let length = usize::try_from(header.payload_len)
        .context("frame payload length does not fit this platform")?;
    let mut payload = vec![0_u8; length];
    recv.read_exact(&mut payload)
        .await
        .context("frame payload ended before its declared length")?;
    let actual = blake3::hash(&payload);
    ensure!(
        actual.as_bytes() == &header.digest,
        "frame payload digest mismatch for request {}",
        header.request_id
    );
    Ok(payload)
}

/// Encode an RPC method and its opaque application payload.
///
/// # Errors
///
/// Returns an error when the method is empty/too long or the combined payload
/// cannot be represented on this platform.
pub fn encode_rpc_request(method: &str, body: &[u8]) -> Result<Vec<u8>> {
    ensure!(!method.is_empty(), "RPC method must not be empty");
    ensure!(
        method.len() <= MAX_RPC_METHOD_LEN,
        "RPC method exceeds {MAX_RPC_METHOD_LEN} bytes"
    );
    let method_len = u16::try_from(method.len()).context("RPC method length overflow")?;
    let capacity = 2_usize
        .checked_add(method.len())
        .and_then(|value| value.checked_add(body.len()))
        .context("RPC request size overflow")?;
    let mut payload = Vec::with_capacity(capacity);
    payload.extend_from_slice(&method_len.to_be_bytes());
    payload.extend_from_slice(method.as_bytes());
    payload.extend_from_slice(body);
    Ok(payload)
}

/// Decode an RPC method and return the remaining opaque body.
///
/// # Errors
///
/// Returns an error for a truncated, empty or non-UTF-8 method.
pub fn decode_rpc_request(payload: &[u8]) -> Result<(&str, &[u8])> {
    ensure!(
        payload.len() >= 2,
        "RPC request is missing its method length"
    );
    let method_len = usize::from(u16::from_be_bytes([payload[0], payload[1]]));
    ensure!(method_len > 0, "RPC method must not be empty");
    ensure!(
        method_len <= MAX_RPC_METHOD_LEN,
        "RPC method exceeds {MAX_RPC_METHOD_LEN} bytes"
    );
    let body_start = 2_usize
        .checked_add(method_len)
        .context("RPC method length overflow")?;
    ensure!(
        payload.len() >= body_start,
        "RPC request ended before its declared method"
    );
    let method =
        std::str::from_utf8(&payload[2..body_start]).context("RPC method is not valid UTF-8")?;
    Ok((method, &payload[body_start..]))
}

fn bounded_chunk_len(remaining: u64, buffer_len: usize) -> usize {
    let buffer_len_u64 = u64::try_from(buffer_len).expect("buffer length fits in u64");
    usize::try_from(remaining.min(buffer_len_u64))
        .expect("bounded length cannot exceed the usize buffer length")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let digest = blake3::hash(b"hello");
        let expected = Header::new(MessageKind::Upload, 42, 5, digest);
        let decoded = Header::decode(&expected.encode()).expect("decode header");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn corrupted_header_is_rejected() {
        let mut encoded = Header::new(MessageKind::Ping, 1, 0, blake3::hash(b"")).encode();
        encoded[0] ^= 1;
        let error = Header::decode(&encoded).expect_err("corruption must fail");
        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn rpc_request_round_trip() {
        let encoded = encode_rpc_request("rpc_pp_forward", b"protobuf").expect("encode RPC");
        let (method, body) = decode_rpc_request(&encoded).expect("decode RPC");
        assert_eq!(method, "rpc_pp_forward");
        assert_eq!(body, b"protobuf");
    }

    #[test]
    fn oversized_rpc_method_is_rejected() {
        let method = "x".repeat(MAX_RPC_METHOD_LEN + 1);
        let error = encode_rpc_request(&method, b"").expect_err("oversized method must fail");
        assert!(error.to_string().contains("exceeds"));
    }
}
