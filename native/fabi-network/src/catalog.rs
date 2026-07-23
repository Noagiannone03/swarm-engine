//! Signed soft-state records for the Fabi swarm discovery catalogue.
//!
//! Protobuf is used as a compact, evolvable container, but its serialization is not canonical.
//! Therefore signatures cover the exact encoded body carried by the envelope. Verifiers never
//! re-encode a parsed body before checking its signature.

use anyhow::{Context, Result, bail, ensure};
use iroh::{EndpointId, SecretKey, Signature};
use prost::Message;

const SIGNING_DOMAIN: &[u8] = b"fabi/swarm/catalog/v3\0";
const KEY_PREFIX: &str = "fabi/swarm/v3";
const PROTOCOL_VERSION: u32 = 3;
const ENDPOINT_ID_BYTES: usize = 32;
const ENDPOINT_ID_HEX: usize = ENDPOINT_ID_BYTES * 2;
const HASH_HEX: usize = 64;

/// Hard DHT limits keep malicious records from becoming memory or bandwidth amplifiers.
pub const MAX_CATALOG_RECORD_BYTES: usize = 32 * 1024;
pub const MAX_CATALOG_PAYLOAD_BYTES: usize = 24 * 1024;
pub const MAX_CATALOG_KEY_BYTES: usize = 256;
pub const MAX_DISCOVERY_PEER_ID_BYTES: usize = 128;
pub const MAX_CATALOG_TTL_MS: u64 = 5 * 60 * 1000;

/// Logical value types stored in the application DHT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum CatalogRecordKind {
    Unspecified = 0,
    ModelManifest = 1,
    WorkerOffer = 2,
    SpanLease = 3,
    LinkMetric = 4,
}

/// Exact bytes signed by the publisher's Iroh endpoint identity.
#[derive(Clone, PartialEq, prost::Message)]
struct CatalogRecordBody {
    #[prost(uint32, tag = "1")]
    protocol_version: u32,
    #[prost(enumeration = "CatalogRecordKind", tag = "2")]
    kind: i32,
    #[prost(string, tag = "3")]
    logical_key: String,
    #[prost(bytes = "vec", tag = "4")]
    publisher_endpoint_id: Vec<u8>,
    #[prost(string, tag = "5")]
    discovery_peer_id: String,
    #[prost(uint64, tag = "6")]
    sequence: u64,
    #[prost(uint64, tag = "7")]
    issued_at_ms: u64,
    #[prost(uint64, tag = "8")]
    expires_at_ms: u64,
    #[prost(bytes = "vec", tag = "9")]
    payload: Vec<u8>,
}

/// Wire envelope kept deliberately small: the signature authenticates `body` byte-for-byte.
#[derive(Clone, PartialEq, prost::Message)]
struct SignedCatalogRecord {
    #[prost(bytes = "vec", tag = "1")]
    body: Vec<u8>,
    #[prost(bytes = "vec", tag = "2")]
    signature: Vec<u8>,
}

/// A record whose size, signature, key binding, protocol, TTL and clock bounds were validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedCatalogRecord {
    pub kind: CatalogRecordKind,
    pub logical_key: String,
    pub publisher_endpoint_id: EndpointId,
    pub discovery_peer_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload: Vec<u8>,
}

/// Parameters for producing one signed catalogue record.
#[derive(Clone, Copy)]
pub struct CatalogRecordParams<'a> {
    pub kind: CatalogRecordKind,
    pub logical_key: &'a str,
    pub discovery_peer_id: &'a str,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload: &'a [u8],
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_logical_key(
    kind: CatalogRecordKind,
    logical_key: &str,
    publisher: &EndpointId,
) -> Result<()> {
    ensure!(
        !logical_key.is_empty() && logical_key.len() <= MAX_CATALOG_KEY_BYTES,
        "catalogue logical key length is invalid"
    );
    ensure!(
        logical_key.is_ascii(),
        "catalogue logical key must be ASCII"
    );

    let parts: Vec<_> = logical_key.split('/').collect();
    let expected_publisher = publisher.to_string();
    match kind {
        CatalogRecordKind::ModelManifest => {
            ensure!(
                parts.len() == 5
                    && parts[..3] == ["fabi", "swarm", "v3"]
                    && parts[3] == "manifest"
                    && is_lower_hex(parts[4], HASH_HEX),
                "invalid model manifest catalogue key"
            );
        }
        CatalogRecordKind::WorkerOffer => {
            ensure!(
                parts.len() == 5
                    && parts[..3] == ["fabi", "swarm", "v3"]
                    && parts[3] == "offer"
                    && parts[4] == expected_publisher,
                "worker offer key is not bound to its publisher"
            );
        }
        CatalogRecordKind::SpanLease => {
            ensure!(
                parts.len() == 6
                    && parts[..3] == ["fabi", "swarm", "v3"]
                    && parts[3] == "span"
                    && is_lower_hex(parts[4], HASH_HEX)
                    && parts[5] == expected_publisher,
                "span lease key is not bound to model and publisher"
            );
        }
        CatalogRecordKind::LinkMetric => {
            ensure!(
                parts.len() == 6
                    && parts[..3] == ["fabi", "swarm", "v3"]
                    && parts[3] == "link"
                    && parts[4] == expected_publisher
                    && is_lower_hex(parts[5], ENDPOINT_ID_HEX),
                "link metric key is not bound to its source publisher"
            );
        }
        CatalogRecordKind::Unspecified => bail!("catalogue record kind is unspecified"),
    }
    Ok(())
}

fn validate_fields(body: &CatalogRecordBody, kind: CatalogRecordKind) -> Result<EndpointId> {
    ensure!(
        body.protocol_version == PROTOCOL_VERSION,
        "unsupported catalogue protocol version {}",
        body.protocol_version
    );
    ensure!(
        !body.discovery_peer_id.is_empty()
            && body.discovery_peer_id.len() <= MAX_DISCOVERY_PEER_ID_BYTES,
        "discovery peer id length is invalid"
    );
    ensure!(
        body.discovery_peer_id.is_ascii(),
        "discovery peer id must be ASCII"
    );
    ensure!(
        !body.payload.is_empty() && body.payload.len() <= MAX_CATALOG_PAYLOAD_BYTES,
        "catalogue payload length is invalid"
    );
    ensure!(
        body.expires_at_ms > body.issued_at_ms,
        "catalogue record must expire after it is issued"
    );
    ensure!(
        body.expires_at_ms - body.issued_at_ms <= MAX_CATALOG_TTL_MS,
        "catalogue record TTL exceeds the protocol maximum"
    );
    let endpoint_bytes: [u8; ENDPOINT_ID_BYTES] = body
        .publisher_endpoint_id
        .as_slice()
        .try_into()
        .context("publisher endpoint id must contain 32 bytes")?;
    let endpoint = EndpointId::from_bytes(&endpoint_bytes)
        .context("publisher endpoint id is not a valid Ed25519 public key")?;
    validate_logical_key(kind, &body.logical_key, &endpoint)?;
    Ok(endpoint)
}

fn signing_preimage(body: &[u8]) -> Vec<u8> {
    let mut preimage = Vec::with_capacity(SIGNING_DOMAIN.len() + body.len());
    preimage.extend_from_slice(SIGNING_DOMAIN);
    preimage.extend_from_slice(body);
    preimage
}

/// Encode and sign one application DHT record with the stable Iroh endpoint key.
///
/// # Errors
///
/// Returns an error when the kind, namespace binding, payload size, or TTL violates the wire
/// contract, or when the final envelope exceeds the DHT record limit.
pub fn sign_catalog_record(
    secret_key: &SecretKey,
    params: CatalogRecordParams<'_>,
) -> Result<Vec<u8>> {
    ensure!(
        params.kind != CatalogRecordKind::Unspecified,
        "catalogue record kind is unspecified"
    );
    let body = CatalogRecordBody {
        protocol_version: PROTOCOL_VERSION,
        kind: params.kind as i32,
        logical_key: params.logical_key.to_owned(),
        publisher_endpoint_id: secret_key.public().as_bytes().to_vec(),
        discovery_peer_id: params.discovery_peer_id.to_owned(),
        sequence: params.sequence,
        issued_at_ms: params.issued_at_ms,
        expires_at_ms: params.expires_at_ms,
        payload: params.payload.to_vec(),
    };
    validate_fields(&body, params.kind)?;

    let body_bytes = body.encode_to_vec();
    let signature = secret_key.sign(&signing_preimage(&body_bytes));
    let encoded = SignedCatalogRecord {
        body: body_bytes,
        signature: signature.to_bytes().to_vec(),
    }
    .encode_to_vec();
    ensure!(
        encoded.len() <= MAX_CATALOG_RECORD_BYTES,
        "signed catalogue record exceeds the protocol maximum"
    );
    Ok(encoded)
}

/// Authenticate and validate a catalogue record before exposing its payload to Python or the DHT.
///
/// # Errors
///
/// Returns an error for malformed or oversized protobuf, an invalid signature or publisher key,
/// a foreign namespace, unsupported protocol data, expiry, or excessive clock skew.
pub fn verify_catalog_record(
    encoded: &[u8],
    now_ms: u64,
    max_clock_skew_ms: u64,
) -> Result<ValidatedCatalogRecord> {
    ensure!(
        !encoded.is_empty() && encoded.len() <= MAX_CATALOG_RECORD_BYTES,
        "signed catalogue record length is invalid"
    );
    let envelope = SignedCatalogRecord::decode(encoded).context("invalid catalogue envelope")?;
    ensure!(
        !envelope.body.is_empty(),
        "catalogue envelope body is empty"
    );
    ensure!(
        envelope.signature.len() == Signature::LENGTH,
        "catalogue signature must contain 64 bytes"
    );

    let body = CatalogRecordBody::decode(envelope.body.as_slice())
        .context("invalid catalogue record body")?;
    let kind = CatalogRecordKind::try_from(body.kind).context("unknown catalogue record kind")?;
    let endpoint = validate_fields(&body, kind)?;
    ensure!(
        body.issued_at_ms <= now_ms.saturating_add(max_clock_skew_ms),
        "catalogue record is issued too far in the future"
    );
    ensure!(body.expires_at_ms > now_ms, "catalogue record has expired");

    let signature = Signature::try_from(envelope.signature.as_slice())
        .context("catalogue signature encoding is invalid")?;
    endpoint
        .verify(&signing_preimage(&envelope.body), &signature)
        .context("catalogue signature verification failed")?;

    Ok(ValidatedCatalogRecord {
        kind,
        logical_key: body.logical_key,
        publisher_endpoint_id: endpoint,
        discovery_peer_id: body.discovery_peer_id,
        sequence: body.sequence,
        issued_at_ms: body.issued_at_ms,
        expires_at_ms: body.expires_at_ms,
        payload: body.payload,
    })
}

/// Canonical logical keys prevent a publisher from writing another worker's mutable namespace.
pub mod keys {
    use super::KEY_PREFIX;
    use iroh::EndpointId;

    #[must_use]
    pub fn manifest(model_swarm_id: &str) -> String {
        format!("{KEY_PREFIX}/manifest/{model_swarm_id}")
    }

    #[must_use]
    pub fn worker_offer(endpoint: &EndpointId) -> String {
        format!("{KEY_PREFIX}/offer/{endpoint}")
    }

    #[must_use]
    pub fn span_lease(model_swarm_id: &str, endpoint: &EndpointId) -> String {
        format!("{KEY_PREFIX}/span/{model_swarm_id}/{endpoint}")
    }

    #[must_use]
    pub fn link_metric(source: &EndpointId, target: &EndpointId) -> String {
        format!("{KEY_PREFIX}/link/{source}/{target}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params<'a>(key: &'a str, payload: &'a [u8]) -> CatalogRecordParams<'a> {
        CatalogRecordParams {
            kind: CatalogRecordKind::WorkerOffer,
            logical_key: key,
            discovery_peer_id: "12D3KooWExample",
            sequence: 7,
            issued_at_ms: 1_000,
            expires_at_ms: 61_000,
            payload,
        }
    }

    #[test]
    fn signed_offer_round_trips_exact_payload_and_identity() {
        let key = SecretKey::generate();
        let logical_key = keys::worker_offer(&key.public());
        let encoded = sign_catalog_record(&key, params(&logical_key, br#"{"worker":"ready"}"#))
            .expect("sign record");

        let record = verify_catalog_record(&encoded, 2_000, 1_000).expect("verify record");
        assert_eq!(record.kind, CatalogRecordKind::WorkerOffer);
        assert_eq!(record.logical_key, logical_key);
        assert_eq!(record.publisher_endpoint_id, key.public());
        assert_eq!(record.sequence, 7);
        assert_eq!(record.payload, br#"{"worker":"ready"}"#);
    }

    #[test]
    fn body_tampering_fails_signature_verification() {
        let key = SecretKey::generate();
        let logical_key = keys::worker_offer(&key.public());
        let encoded =
            sign_catalog_record(&key, params(&logical_key, b"original")).expect("sign record");
        let mut envelope = SignedCatalogRecord::decode(encoded.as_slice()).expect("decode");
        let last = envelope.body.last_mut().expect("nonempty body");
        *last ^= 1;

        let error = verify_catalog_record(&envelope.encode_to_vec(), 2_000, 1_000)
            .expect_err("tampering must fail");
        assert!(error.to_string().contains("signature verification failed"));
    }

    #[test]
    fn publisher_cannot_sign_into_another_workers_namespace() {
        let publisher = SecretKey::generate();
        let victim = SecretKey::generate();
        let logical_key = keys::worker_offer(&victim.public());
        let error = sign_catalog_record(&publisher, params(&logical_key, b"payload"))
            .expect_err("foreign namespace must fail");
        assert!(error.to_string().contains("not bound to its publisher"));
    }

    #[test]
    fn expired_future_and_excessive_ttl_records_fail_closed() {
        let key = SecretKey::generate();
        let logical_key = keys::worker_offer(&key.public());

        let expired = sign_catalog_record(&key, params(&logical_key, b"payload"))
            .expect("sign expired fixture");
        assert!(
            verify_catalog_record(&expired, 61_000, 0)
                .expect_err("expiry is exclusive")
                .to_string()
                .contains("expired")
        );

        let mut future = params(&logical_key, b"payload");
        future.issued_at_ms = 10_000;
        future.expires_at_ms = 20_000;
        let future = sign_catalog_record(&key, future).expect("sign future fixture");
        assert!(
            verify_catalog_record(&future, 1_000, 1_000)
                .expect_err("future issue time must fail")
                .to_string()
                .contains("future")
        );

        let mut excessive = params(&logical_key, b"payload");
        excessive.expires_at_ms = excessive.issued_at_ms + MAX_CATALOG_TTL_MS + 1;
        assert!(
            sign_catalog_record(&key, excessive)
                .expect_err("excessive TTL must fail")
                .to_string()
                .contains("TTL")
        );
    }

    #[test]
    fn all_mutable_key_types_bind_the_publisher() {
        let source = SecretKey::generate();
        let target = SecretKey::generate();
        let model = "a".repeat(64);
        let cases = [
            (
                CatalogRecordKind::SpanLease,
                keys::span_lease(&model, &source.public()),
            ),
            (
                CatalogRecordKind::LinkMetric,
                keys::link_metric(&source.public(), &target.public()),
            ),
        ];
        for (kind, logical_key) in cases {
            let mut record_params = params(&logical_key, b"payload");
            record_params.kind = kind;
            let encoded = sign_catalog_record(&source, record_params).expect("sign bound key");
            assert_eq!(
                verify_catalog_record(&encoded, 2_000, 0)
                    .expect("verify bound key")
                    .kind,
                kind
            );
        }
    }
}
