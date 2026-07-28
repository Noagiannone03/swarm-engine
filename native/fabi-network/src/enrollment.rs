//! Endpoint-owned relay enrollment proofs.
//!
//! The public registry authenticates the Fabi account credential separately.
//! This module proves possession of the stable Iroh endpoint key without ever
//! exporting it from the native transport boundary.

use anyhow::{Context, Result, ensure};
use data_encoding::HEXLOWER;
use iroh::{EndpointId, SecretKey};

const ENROLLMENT_DOMAIN: &[u8] = b"fabi/network/relay-enrollment/v1\0";

fn decode_32_hex(value: &str, name: &str) -> Result<[u8; 32]> {
    ensure!(
        value.len() == 64,
        "{name} must contain 64 hexadecimal characters"
    );
    let bytes = HEXLOWER
        .decode(value.as_bytes())
        .with_context(|| format!("{name} must be lowercase hexadecimal"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("{name} decoded to an invalid length"))
}

fn signing_preimage(
    endpoint: &EndpointId,
    account_id: &str,
    issued_at_ms: u64,
    nonce: &str,
) -> Result<Vec<u8>> {
    let account = decode_32_hex(account_id, "account ID")?;
    let nonce = decode_32_hex(nonce, "enrollment nonce")?;
    let mut message = Vec::with_capacity(ENROLLMENT_DOMAIN.len() + 32 + 32 + 8 + 32);
    message.extend_from_slice(ENROLLMENT_DOMAIN);
    message.extend_from_slice(endpoint.as_bytes());
    message.extend_from_slice(&account);
    message.extend_from_slice(&issued_at_ms.to_be_bytes());
    message.extend_from_slice(&nonce);
    Ok(message)
}

/// Sign a domain-separated, bounded relay enrollment proof.
///
/// The endpoint public key is included in the signed preimage, as are the
/// server-derived account hash, timestamp and one-use nonce.
///
/// # Errors
///
/// Returns an error when the account identifier or nonce is not exactly a
/// lowercase hexadecimal 32-byte value.
pub fn sign(
    secret_key: &SecretKey,
    account_id: &str,
    issued_at_ms: u64,
    nonce: &str,
) -> Result<[u8; 64]> {
    let endpoint = secret_key.public();
    Ok(secret_key
        .sign(&signing_preimage(
            &endpoint,
            account_id,
            issued_at_ms,
            nonce,
        )?)
        .to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::Signature;

    #[test]
    fn proof_is_bound_to_endpoint_account_time_and_nonce() -> Result<()> {
        let key = SecretKey::generate();
        let account = "ab".repeat(32);
        let nonce = "cd".repeat(32);
        let signature = Signature::from_bytes(&sign(&key, &account, 1_800_000_000_000, &nonce)?);
        key.public().verify(
            &signing_preimage(&key.public(), &account, 1_800_000_000_000, &nonce)?,
            &signature,
        )?;

        assert!(
            key.public()
                .verify(
                    &signing_preimage(&key.public(), &"ef".repeat(32), 1_800_000_000_000, &nonce)?,
                    &signature,
                )
                .is_err()
        );
        assert!(sign(&key, &account, 0, "NOT-LOWERCASE").is_err());
        Ok(())
    }
}
