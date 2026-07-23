//! Signed soft-state records for the Fabi swarm discovery catalogue.
//!
//! Protobuf is used as a compact, evolvable container, but its serialization is not canonical.
//! Therefore signatures cover the exact encoded body carried by the envelope. Verifiers never
//! re-encode a parsed body before checking its signature.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use iroh::{EndpointId, SecretKey, Signature};
use prost::Message;

const SIGNING_DOMAIN: &[u8] = b"fabi/swarm/catalog/v3\0";
const KEY_PREFIX: &str = "fabi/swarm/v3";
const PROTOCOL_VERSION: u32 = 3;
const ENDPOINT_ID_BYTES: usize = 32;
const ENDPOINT_ID_HEX: usize = ENDPOINT_ID_BYTES * 2;
const HASH_HEX: usize = 64;
const CATALOG_SET_MAGIC: &[u8] = b"FABISET3\0";

/// Fixed protocol-level fan-out for model membership.  A worker has exactly one shard per model,
/// derived from its authenticated endpoint id, so it cannot choose a hot or misleading shard.
pub const MODEL_MEMBERSHIP_SHARDS: u16 = 64;

/// Hard DHT limits keep malicious records from becoming memory or bandwidth amplifiers.
pub const MAX_CATALOG_RECORD_BYTES: usize = 32 * 1024;
/// Membership sets are larger than individual records but remain bounded for Kademlia packets.
pub const MAX_CATALOG_SET_BYTES: usize = 256 * 1024;
pub const MAX_CATALOG_SET_ENTRIES: usize = 512;
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
    ModelMember = 5,
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

/// Hivemind-style independently expiring subkeys, encoded as one Kademlia value.
///
/// The container is not trusted or signed. Every opaque entry carries its own endpoint signature,
/// sequence, key binding and TTL. Routing nodes merge valid entries and discard invalid ones.
#[derive(Clone, PartialEq, prost::Message)]
struct CatalogRecordSet {
    #[prost(uint32, tag = "1")]
    protocol_version: u32,
    #[prost(string, tag = "2")]
    logical_key: String,
    #[prost(bytes = "vec", repeated, tag = "3")]
    entries: Vec<Vec<u8>>,
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
        CatalogRecordKind::ModelMember => {
            ensure!(
                parts.len() == 6
                    && parts[..3] == ["fabi", "swarm", "v3"]
                    && parts[3] == "member"
                    && is_lower_hex(parts[4], HASH_HEX)
                    && parts[5] == format!("{:02x}", membership_shard(publisher)),
                "model membership key is not bound to its publisher shard"
            );
        }
        CatalogRecordKind::Unspecified => bail!("catalogue record kind is unspecified"),
    }
    Ok(())
}

/// Return the deterministic model-membership shard for an authenticated endpoint.
#[must_use]
pub fn membership_shard(endpoint: &EndpointId) -> u16 {
    let digest = blake3::hash(endpoint.as_bytes());
    u16::from_be_bytes([digest.as_bytes()[0], digest.as_bytes()[1]]) % MODEL_MEMBERSHIP_SHARDS
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

fn decode_catalog_set(encoded: &[u8]) -> Result<CatalogRecordSet> {
    ensure!(
        encoded.starts_with(CATALOG_SET_MAGIC),
        "catalogue value is not a membership set"
    );
    ensure!(
        encoded.len() <= MAX_CATALOG_SET_BYTES,
        "catalogue membership set exceeds the protocol maximum"
    );
    let set = CatalogRecordSet::decode(&encoded[CATALOG_SET_MAGIC.len()..])
        .context("invalid catalogue membership set")?;
    ensure!(
        set.protocol_version == PROTOCOL_VERSION,
        "unsupported catalogue membership set version {}",
        set.protocol_version
    );
    ensure!(
        !set.entries.is_empty() && set.entries.len() <= MAX_CATALOG_SET_ENTRIES,
        "catalogue membership set entry count is invalid"
    );
    Ok(set)
}

fn verified_membership_entries(
    expected_key: &str,
    encoded: &[u8],
    now_ms: u64,
    max_clock_skew_ms: u64,
) -> Result<Vec<(ValidatedCatalogRecord, Vec<u8>)>> {
    let candidates = if encoded.starts_with(CATALOG_SET_MAGIC) {
        let set = decode_catalog_set(encoded)?;
        ensure!(
            set.logical_key == expected_key,
            "membership set Kademlia key does not match its logical key"
        );
        set.entries
    } else {
        vec![encoded.to_vec()]
    };

    let mut valid = Vec::with_capacity(candidates.len());
    for entry in candidates {
        let Ok(record) = verify_catalog_record(&entry, now_ms, max_clock_skew_ms) else {
            // One malicious or expired subkey must not poison unrelated publishers in the set.
            continue;
        };
        if record.kind == CatalogRecordKind::ModelMember && record.logical_key == expected_key {
            valid.push((record, entry));
        }
    }
    ensure!(
        !valid.is_empty(),
        "catalogue membership value contains no valid live entries"
    );
    Ok(valid)
}

fn prefer_membership_candidate(
    previous: &(ValidatedCatalogRecord, Vec<u8>),
    candidate: &(ValidatedCatalogRecord, Vec<u8>),
) -> bool {
    if candidate.0.sequence != previous.0.sequence {
        return candidate.0.sequence > previous.0.sequence;
    }
    if candidate.1 == previous.1 {
        return false;
    }
    // A publisher equivocated at one sequence. Pick a deterministic byte ordering so replicas
    // converge independent of arrival order; higher layers can quarantine the signed evidence.
    blake3::hash(&candidate.1).as_bytes() < blake3::hash(&previous.1).as_bytes()
}

/// Merge one or more membership deltas/snapshots into a deterministic signed-entry set.
///
/// This is the native equivalent of Hivemind dictionary subkeys: publishers update only their own
/// independently signed entry, while DHT replicas converge by publisher and sequence.
///
/// # Errors
///
/// Returns an error when no input contains a valid live member, the shard is full, or the merged
/// representation exceeds its wire bound.
pub fn merge_catalog_membership_values<'a>(
    expected_key: &str,
    values: impl IntoIterator<Item = &'a [u8]>,
    now_ms: u64,
    max_clock_skew_ms: u64,
) -> Result<Vec<u8>> {
    let mut entries: BTreeMap<String, (ValidatedCatalogRecord, Vec<u8>)> = BTreeMap::new();
    for encoded in values {
        let Ok(candidates) =
            verified_membership_entries(expected_key, encoded, now_ms, max_clock_skew_ms)
        else {
            continue;
        };
        for candidate in candidates {
            let publisher = candidate.0.publisher_endpoint_id.to_string();
            match entries.get(&publisher) {
                Some(previous) if !prefer_membership_candidate(previous, &candidate) => {}
                _ => {
                    entries.insert(publisher, candidate);
                }
            }
        }
    }
    ensure!(
        !entries.is_empty(),
        "cannot encode an empty catalogue membership set"
    );
    ensure!(
        entries.len() <= MAX_CATALOG_SET_ENTRIES,
        "catalogue membership shard is full"
    );
    let set = CatalogRecordSet {
        protocol_version: PROTOCOL_VERSION,
        logical_key: expected_key.to_owned(),
        entries: entries.into_values().map(|(_, encoded)| encoded).collect(),
    };
    let mut encoded = Vec::with_capacity(CATALOG_SET_MAGIC.len() + set.encoded_len());
    encoded.extend_from_slice(CATALOG_SET_MAGIC);
    set.encode(&mut encoded)
        .context("failed to encode catalogue membership set")?;
    ensure!(
        encoded.len() <= MAX_CATALOG_SET_BYTES,
        "catalogue membership set exceeds the protocol maximum"
    );
    Ok(encoded)
}

/// Verify and expose all live independently signed members from one DHT value.
///
/// # Errors
///
/// Returns an error when the value is malformed, belongs to another shard, or contains no valid
/// live signed entry.
pub fn verify_catalog_membership_value(
    expected_key: &str,
    encoded: &[u8],
    now_ms: u64,
    max_clock_skew_ms: u64,
) -> Result<Vec<ValidatedCatalogRecord>> {
    let merged =
        merge_catalog_membership_values(expected_key, [encoded], now_ms, max_clock_skew_ms)?;
    Ok(
        verified_membership_entries(expected_key, &merged, now_ms, max_clock_skew_ms)?
            .into_iter()
            .map(|(record, _)| record)
            .collect(),
    )
}

/// Canonical logical keys prevent a publisher from writing another worker's mutable namespace.
pub mod keys {
    use super::{KEY_PREFIX, membership_shard};
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

    #[must_use]
    pub fn model_member(model_swarm_id: &str, endpoint: &EndpointId) -> String {
        format!(
            "{KEY_PREFIX}/member/{model_swarm_id}/{:02x}",
            membership_shard(endpoint)
        )
    }

    #[must_use]
    pub fn all_model_membership_shards(model_swarm_id: &str) -> Vec<String> {
        (0..super::MODEL_MEMBERSHIP_SHARDS)
            .map(|shard| format!("{KEY_PREFIX}/member/{model_swarm_id}/{shard:02x}"))
            .collect()
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

    fn signed_member(key: &SecretKey, model: &str, sequence: u64, issued_at_ms: u64) -> Vec<u8> {
        let logical_key = keys::model_member(model, &key.public());
        sign_catalog_record(
            key,
            CatalogRecordParams {
                kind: CatalogRecordKind::ModelMember,
                logical_key: &logical_key,
                discovery_peer_id: "12D3KooWMember",
                sequence,
                issued_at_ms,
                expires_at_ms: issued_at_ms + 60_000,
                payload: b"member",
            },
        )
        .expect("sign member")
    }

    #[test]
    fn membership_shards_are_publisher_bound_and_sets_merge_independent_entries() {
        let model = "a".repeat(HASH_HEX);
        let first = SecretKey::generate();
        let shard = membership_shard(&first.public());
        let second = (0..10_000)
            .map(|_| SecretKey::generate())
            .find(|key| membership_shard(&key.public()) == shard)
            .expect("find a second endpoint in the same shard");
        let logical_key = keys::model_member(&model, &first.public());
        assert_eq!(logical_key, keys::model_member(&model, &second.public()));

        let first_record = signed_member(&first, &model, 1, 1_000);
        let second_record = signed_member(&second, &model, 3, 1_000);
        let set = merge_catalog_membership_values(
            &logical_key,
            [&first_record[..], &second_record[..]],
            2_000,
            0,
        )
        .expect("merge members");
        let members =
            verify_catalog_membership_value(&logical_key, &set, 2_000, 0).expect("verify members");
        assert_eq!(members.len(), 2);
        assert!(members.windows(2).all(|pair| {
            pair[0].publisher_endpoint_id.to_string() < pair[1].publisher_endpoint_id.to_string()
        }));

        let wrong_shard = format!(
            "{KEY_PREFIX}/member/{model}/{:02x}",
            (shard + 1) % MODEL_MEMBERSHIP_SHARDS
        );
        let error = sign_catalog_record(
            &first,
            CatalogRecordParams {
                kind: CatalogRecordKind::ModelMember,
                logical_key: &wrong_shard,
                discovery_peer_id: "12D3KooWMember",
                sequence: 1,
                issued_at_ms: 1_000,
                expires_at_ms: 61_000,
                payload: b"member",
            },
        )
        .expect_err("publisher must not choose a membership shard");
        assert!(error.to_string().contains("publisher shard"));
    }

    #[test]
    fn membership_merge_converges_and_keeps_the_highest_sequence() {
        let model = "b".repeat(HASH_HEX);
        let publisher = SecretKey::generate();
        let logical_key = keys::model_member(&model, &publisher.public());
        let old = signed_member(&publisher, &model, 4, 1_000);
        let new = signed_member(&publisher, &model, 9, 1_100);

        let forward = merge_catalog_membership_values(&logical_key, [&old[..], &new[..]], 2_000, 0)
            .expect("merge forward");
        let reverse = merge_catalog_membership_values(&logical_key, [&new[..], &old[..]], 2_000, 0)
            .expect("merge reverse");
        assert_eq!(forward, reverse);
        assert_eq!(
            verify_catalog_membership_value(&logical_key, &forward, 2_000, 0)
                .expect("verify merged")
                .pop()
                .expect("one member")
                .sequence,
            9
        );
    }
}
