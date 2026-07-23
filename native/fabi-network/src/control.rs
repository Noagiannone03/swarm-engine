//! Domain-separated signatures for protocol-v3 control messages.
//!
//! Iroh authenticates the live QUIC connection. These signatures additionally
//! make route plans and reservation receipts portable, auditable records that
//! remain bound to the stable endpoint identity after the RPC completes.

use anyhow::{Context, Result, ensure};
use iroh::{EndpointId, SecretKey, Signature};

const CONTROL_SIGNING_DOMAIN: &[u8] = b"fabi/swarm/control/v3\0";
pub const MAX_CONTROL_PAYLOAD_BYTES: usize = 256 * 1024;

fn signing_preimage(payload: &[u8]) -> Result<Vec<u8>> {
    ensure!(!payload.is_empty(), "control payload is empty");
    ensure!(
        payload.len() <= MAX_CONTROL_PAYLOAD_BYTES,
        "control payload exceeds the protocol maximum"
    );
    let mut preimage = Vec::with_capacity(CONTROL_SIGNING_DOMAIN.len() + payload.len());
    preimage.extend_from_slice(CONTROL_SIGNING_DOMAIN);
    preimage.extend_from_slice(payload);
    Ok(preimage)
}

/// Sign exact protocol-v3 control bytes with the stable Iroh endpoint key.
///
/// # Errors
///
/// Returns an error when the payload is empty or exceeds the bounded control
/// message size.
pub fn sign_control_payload(secret_key: &SecretKey, payload: &[u8]) -> Result<[u8; 64]> {
    let signature = secret_key.sign(&signing_preimage(payload)?);
    Ok(signature.to_bytes())
}

/// Verify exact protocol-v3 control bytes against an expected endpoint.
///
/// # Errors
///
/// Returns an error for malformed signatures, oversized payloads, or a failed
/// Ed25519 verification.
pub fn verify_control_payload(
    endpoint: &EndpointId,
    payload: &[u8],
    signature: &[u8],
) -> Result<()> {
    ensure!(
        signature.len() == Signature::LENGTH,
        "control signature must contain 64 bytes"
    );
    let signature =
        Signature::try_from(signature).context("control signature encoding is invalid")?;
    endpoint
        .verify(&signing_preimage(payload)?, &signature)
        .context("control signature verification failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_signature_binds_payload_and_endpoint() {
        let signer = SecretKey::generate();
        let other = SecretKey::generate();
        let payload = br#"{"route_id":"route-1","epoch":7}"#;
        let signature = sign_control_payload(&signer, payload).expect("sign");

        verify_control_payload(&signer.public(), payload, &signature).expect("verify");
        assert!(verify_control_payload(&other.public(), payload, &signature).is_err());
        assert!(
            verify_control_payload(&signer.public(), b"{\"route_id\":\"route-2\"}", &signature)
                .is_err()
        );
    }

    #[test]
    fn control_signature_rejects_unbounded_payloads() {
        let signer = SecretKey::generate();
        assert!(sign_control_payload(&signer, &[]).is_err());
        assert!(sign_control_payload(&signer, &vec![0; MAX_CONTROL_PAYLOAD_BYTES + 1]).is_err());
    }
}
