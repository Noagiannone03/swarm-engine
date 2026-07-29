//! Short-lived, route-bound capabilities for client-coordinated inference.
//!
//! A worker must not trust a permanently configured coordinator `EndpointId`.
//! Instead, the Fabi admission authority issues one sealed Biscuit bound to the
//! exact request, model, coordinator, route-plan digest and epoch. Workers only
//! need the authority public key to verify it; prompts and generated tokens
//! never enter the authorization service.

use std::{
    collections::BTreeSet,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use biscuit_auth::{
    Biscuit, KeyPair, PublicKey,
    builder::Algorithm,
    macros::{authorizer, biscuit},
};
use data_encoding::HEXLOWER;
use serde::{Deserialize, Serialize};

const HASH_HEX_LEN: usize = 64;
const MAX_TEXT_LEN: usize = 512;
const MAX_CAPABILITY_LIFETIME_MS: u64 = 5 * 60 * 1_000;
const MAX_CLOCK_SKEW_MS: u64 = 30_000;

/// Explicit recovery cost/guarantee attached to an admitted route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteRecoveryPolicy {
    BestEffort,
    ReplanCold,
    ActivationReplay,
    ReservedRoute,
    HotReplica,
}

impl RouteRecoveryPolicy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::BestEffort => "best_effort",
            Self::ReplanCold => "replan_cold",
            Self::ActivationReplay => "activation_replay",
            Self::ReservedRoute => "reserved_route",
            Self::HotReplica => "hot_replica",
        }
    }
}

/// Authority-signed rights for exactly one route generation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCapabilityClaims {
    pub permit_id: String,
    pub account_id: String,
    pub request_id: String,
    pub model_swarm_id: String,
    pub coordinator_endpoint_id: String,
    pub route_plan_digest: String,
    pub epoch: u64,
    #[serde(default)]
    pub authorization_generation: u64,
    pub max_context_tokens: u64,
    pub recovery_policy: RouteRecoveryPolicy,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Ambient request facts supplied by a worker during verification.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCapabilityContext {
    pub permit_id: String,
    pub account_id: String,
    pub request_id: String,
    pub model_swarm_id: String,
    pub coordinator_endpoint_id: String,
    pub route_plan_digest: String,
    pub epoch: u64,
    #[serde(default)]
    pub authorization_generation: u64,
    pub capability_expires_at_ms: u64,
    pub required_context_tokens: u64,
    pub recovery_policy: RouteRecoveryPolicy,
    pub now_ms: u64,
}

/// A successful verification result suitable for idempotency and revocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedRouteCapability {
    pub permit_id: String,
    pub root_revocation_id: Vec<u8>,
}

fn validate_text(value: &str, field: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{field} must not be empty");
    ensure!(
        value.len() <= MAX_TEXT_LEN,
        "{field} exceeds {MAX_TEXT_LEN} bytes"
    );
    ensure!(
        !value.chars().any(char::is_control),
        "{field} contains control characters"
    );
    Ok(())
}

fn validate_hash(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == HASH_HEX_LEN,
        "{field} must contain {HASH_HEX_LEN} lowercase hexadecimal characters"
    );
    HEXLOWER
        .decode(value.as_bytes())
        .with_context(|| format!("{field} must be lowercase hexadecimal"))?;
    Ok(())
}

fn checked_i64(value: u64, field: &str) -> Result<i64> {
    i64::try_from(value).with_context(|| format!("{field} exceeds the Biscuit integer range"))
}

fn system_time(milliseconds: u64, field: &str) -> Result<SystemTime> {
    UNIX_EPOCH
        .checked_add(Duration::from_millis(milliseconds))
        .with_context(|| format!("{field} exceeds the system clock range"))
}

fn validate_claims(claims: &RouteCapabilityClaims) -> Result<()> {
    validate_hash(&claims.permit_id, "permit ID")?;
    validate_hash(&claims.account_id, "account ID")?;
    validate_text(&claims.request_id, "request ID")?;
    validate_hash(&claims.model_swarm_id, "model swarm ID")?;
    validate_hash(&claims.coordinator_endpoint_id, "coordinator EndpointId")?;
    validate_hash(&claims.route_plan_digest, "route plan digest")?;
    ensure!(claims.epoch > 0, "route capability epoch must be positive");
    ensure!(
        claims.max_context_tokens > 0,
        "route capability context budget must be positive"
    );
    ensure!(
        claims.expires_at_ms > claims.issued_at_ms,
        "route capability must expire after issuance"
    );
    ensure!(
        claims.expires_at_ms - claims.issued_at_ms <= MAX_CAPABILITY_LIFETIME_MS,
        "route capability lifetime exceeds {MAX_CAPABILITY_LIFETIME_MS} ms"
    );
    checked_i64(claims.epoch, "epoch")?;
    checked_i64(claims.authorization_generation, "authorization generation")?;
    checked_i64(claims.max_context_tokens, "context budget")?;
    checked_i64(claims.expires_at_ms, "capability expiry")?;
    Ok(())
}

fn validate_context(context: &RouteCapabilityContext) -> Result<()> {
    validate_hash(&context.permit_id, "permit ID")?;
    validate_hash(&context.account_id, "account ID")?;
    validate_text(&context.request_id, "request ID")?;
    validate_hash(&context.model_swarm_id, "model swarm ID")?;
    validate_hash(&context.coordinator_endpoint_id, "coordinator EndpointId")?;
    validate_hash(&context.route_plan_digest, "route plan digest")?;
    ensure!(context.epoch > 0, "route capability epoch must be positive");
    ensure!(
        context.capability_expires_at_ms > context.now_ms,
        "route capability has expired"
    );
    ensure!(
        context.required_context_tokens > 0,
        "required context tokens must be positive"
    );
    checked_i64(context.epoch, "epoch")?;
    checked_i64(context.authorization_generation, "authorization generation")?;
    checked_i64(context.capability_expires_at_ms, "capability expiry")?;
    checked_i64(context.required_context_tokens, "required context tokens")?;
    Ok(())
}

fn ed25519_keypair(private_key_hex: &str) -> Result<KeyPair> {
    let bytes = HEXLOWER
        .decode(private_key_hex.as_bytes())
        .context("capability private key must be lowercase hexadecimal")?;
    KeyPair::from_bytes(&bytes, Algorithm::Ed25519.into())
        .context("invalid Ed25519 capability private key")
}

fn ed25519_public_key(public_key_hex: &str) -> Result<PublicKey> {
    let bytes = HEXLOWER
        .decode(public_key_hex.as_bytes())
        .context("capability public key must be lowercase hexadecimal")?;
    PublicKey::from_bytes(&bytes, Algorithm::Ed25519)
        .context("invalid Ed25519 capability public key")
}

/// Issue one sealed route capability.
///
/// Sealing prevents a holder from appending arbitrary blocks. Route changes or
/// recovery epochs require a fresh authority decision and therefore cannot
/// silently widen the original admission.
///
/// # Errors
///
/// Returns an error for malformed claims, keys or Biscuit serialization.
pub fn issue_route_capability(
    private_key_hex: &str,
    claims: &RouteCapabilityClaims,
) -> Result<String> {
    validate_claims(claims)?;
    let root = ed25519_keypair(private_key_hex)?;
    let epoch = checked_i64(claims.epoch, "epoch")?;
    let authorization_generation =
        checked_i64(claims.authorization_generation, "authorization generation")?;
    let max_context_tokens = checked_i64(claims.max_context_tokens, "context budget")?;
    let expires_at_ms = checked_i64(claims.expires_at_ms, "capability expiry")?;
    let not_before = system_time(
        claims.issued_at_ms.saturating_sub(MAX_CLOCK_SKEW_MS),
        "capability not-before time",
    )?;
    let expiration = system_time(claims.expires_at_ms, "capability expiry")?;

    biscuit!(
        r#"
        fabi_route_permit(
            {permit_id},
            {account_id},
            {request_id},
            {model_swarm_id},
            {coordinator_endpoint_id},
            {route_plan_digest},
            {epoch},
            {authorization_generation},
            {expires_at_ms},
            {max_context_tokens},
            {recovery_policy}
        );
        check if time($time), $time >= {not_before}, $time < {expiration};
        "#,
        permit_id = claims.permit_id.as_str(),
        account_id = claims.account_id.as_str(),
        request_id = claims.request_id.as_str(),
        model_swarm_id = claims.model_swarm_id.as_str(),
        coordinator_endpoint_id = claims.coordinator_endpoint_id.as_str(),
        route_plan_digest = claims.route_plan_digest.as_str(),
        authorization_generation = authorization_generation,
        expires_at_ms = expires_at_ms,
        recovery_policy = claims.recovery_policy.as_str(),
    )
    .build(&root)
    .context("failed to build route capability")?
    .seal()
    .context("failed to seal route capability")?
    .to_base64()
    .context("failed to serialize route capability")
}

/// Verify a route capability against worker-observed ambient facts.
///
/// The caller supplies the exact route-plan digest and coordinator identity
/// already authenticated by the Iroh connection. A copied token therefore
/// cannot authorize a different coordinator, plan, model, epoch or context.
///
/// # Errors
///
/// Returns an error when signature, expiry, revocation or any ambient fact does
/// not match.
pub fn verify_route_capability(
    public_key_hex: &str,
    token: &str,
    context: &RouteCapabilityContext,
    revoked_identifiers: &BTreeSet<Vec<u8>>,
) -> Result<VerifiedRouteCapability> {
    validate_context(context)?;
    let root = ed25519_public_key(public_key_hex)?;
    let biscuit = Biscuit::from_base64(token, root).context("invalid signed route capability")?;
    let revocation_identifiers = biscuit.revocation_identifiers();
    ensure!(
        !revocation_identifiers
            .iter()
            .any(|identifier| revoked_identifiers.contains(identifier)),
        "route capability has been revoked"
    );

    let epoch = checked_i64(context.epoch, "epoch")?;
    let authorization_generation =
        checked_i64(context.authorization_generation, "authorization generation")?;
    let capability_expires_at_ms =
        checked_i64(context.capability_expires_at_ms, "capability expiry")?;
    let required_context_tokens =
        checked_i64(context.required_context_tokens, "required context tokens")?;
    let now = system_time(context.now_ms, "verification time")?;
    let mut authorizer = authorizer!(
        r#"
        time({now});
        fabi_route_request(
            {permit_id},
            {account_id},
            {request_id},
            {model_swarm_id},
            {coordinator_endpoint_id},
            {route_plan_digest},
            {epoch},
            {authorization_generation},
            {capability_expires_at_ms},
            {required_context_tokens},
            {recovery_policy}
        );
        allow if
            fabi_route_permit(
                $permit_id,
                $account_id,
                $request_id,
                $model_swarm_id,
                $coordinator_endpoint_id,
                $route_plan_digest,
                $epoch,
                $authorization_generation,
                $capability_expires_at_ms,
                $max_context_tokens,
                $recovery_policy
            ),
            fabi_route_request(
                $permit_id,
                $account_id,
                $request_id,
                $model_swarm_id,
                $coordinator_endpoint_id,
                $route_plan_digest,
                $epoch,
                $authorization_generation,
                $capability_expires_at_ms,
                $required_context_tokens,
                $recovery_policy
            ),
            $required_context_tokens <= $max_context_tokens;
        deny if true;
        "#,
        permit_id = context.permit_id.as_str(),
        account_id = context.account_id.as_str(),
        request_id = context.request_id.as_str(),
        model_swarm_id = context.model_swarm_id.as_str(),
        coordinator_endpoint_id = context.coordinator_endpoint_id.as_str(),
        route_plan_digest = context.route_plan_digest.as_str(),
        authorization_generation = authorization_generation,
        capability_expires_at_ms = capability_expires_at_ms,
        recovery_policy = context.recovery_policy.as_str(),
    )
    .build(&biscuit)
    .context("failed to construct route capability authorizer")?;
    authorizer
        .authorize()
        .context("route capability does not authorize this plan")?;

    let root_revocation_id = revocation_identifiers
        .into_iter()
        .next()
        .context("route capability has no revocation identifier")?;
    Ok(VerifiedRouteCapability {
        permit_id: context.permit_id.clone(),
        root_revocation_id,
    })
}

/// Return the authority block revocation identifier of a signed capability.
///
/// This authenticates the token before exposing the identifier that the
/// issuance ledger persists for emergency revocation.
///
/// # Errors
///
/// Returns an error for an invalid signature or malformed token.
pub fn route_capability_root_revocation_id(public_key_hex: &str, token: &str) -> Result<Vec<u8>> {
    let root = ed25519_public_key(public_key_hex)?;
    let biscuit = Biscuit::from_base64(token, root).context("invalid signed route capability")?;
    biscuit
        .revocation_identifiers()
        .first()
        .cloned()
        .context("route capability has no authority revocation identifier")
}

/// Return the Ed25519 public key for a capability issuer private key.
///
/// # Errors
///
/// Returns an error for malformed private key material.
pub fn capability_public_key(private_key_hex: &str) -> Result<String> {
    Ok(ed25519_keypair(private_key_hex)?.public().to_bytes_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims(now_ms: u64) -> RouteCapabilityClaims {
        RouteCapabilityClaims {
            permit_id: "11".repeat(32),
            account_id: "22".repeat(32),
            request_id: "request-7".to_owned(),
            model_swarm_id: "33".repeat(32),
            coordinator_endpoint_id: "44".repeat(32),
            route_plan_digest: "55".repeat(32),
            epoch: 7,
            authorization_generation: 0,
            max_context_tokens: 16_384,
            recovery_policy: RouteRecoveryPolicy::ReplanCold,
            issued_at_ms: now_ms,
            expires_at_ms: now_ms + 60_000,
        }
    }

    fn context(claims: &RouteCapabilityClaims, now_ms: u64) -> RouteCapabilityContext {
        RouteCapabilityContext {
            permit_id: claims.permit_id.clone(),
            account_id: claims.account_id.clone(),
            request_id: claims.request_id.clone(),
            model_swarm_id: claims.model_swarm_id.clone(),
            coordinator_endpoint_id: claims.coordinator_endpoint_id.clone(),
            route_plan_digest: claims.route_plan_digest.clone(),
            epoch: claims.epoch,
            authorization_generation: claims.authorization_generation,
            capability_expires_at_ms: claims.expires_at_ms,
            required_context_tokens: 12_000,
            recovery_policy: claims.recovery_policy,
            now_ms,
        }
    }

    fn issuer() -> (String, String) {
        let keypair = KeyPair::new();
        (
            keypair.private().to_bytes_hex(),
            keypair.public().to_bytes_hex(),
        )
    }

    #[test]
    fn capability_is_bound_to_route_coordinator_epoch_and_context() -> Result<()> {
        let now_ms = 1_800_000_000_000;
        let (private_key, public_key) = issuer();
        let claims = claims(now_ms);
        let token = issue_route_capability(&private_key, &claims)?;
        let expected = context(&claims, now_ms + 1_000);

        let verified = verify_route_capability(&public_key, &token, &expected, &BTreeSet::new())?;
        assert_eq!(verified.permit_id, claims.permit_id);
        assert!(!verified.root_revocation_id.is_empty());

        for rejected in [
            RouteCapabilityContext {
                coordinator_endpoint_id: "66".repeat(32),
                ..expected.clone()
            },
            RouteCapabilityContext {
                route_plan_digest: "77".repeat(32),
                ..expected.clone()
            },
            RouteCapabilityContext {
                epoch: expected.epoch + 1,
                ..expected.clone()
            },
            RouteCapabilityContext {
                authorization_generation: expected.authorization_generation + 1,
                ..expected.clone()
            },
            RouteCapabilityContext {
                capability_expires_at_ms: expected.capability_expires_at_ms + 1,
                ..expected.clone()
            },
            RouteCapabilityContext {
                required_context_tokens: claims.max_context_tokens + 1,
                ..expected
            },
        ] {
            assert!(
                verify_route_capability(&public_key, &token, &rejected, &BTreeSet::new()).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn expired_and_revoked_capabilities_fail_closed() -> Result<()> {
        let now_ms = 1_800_000_000_000;
        let (private_key, public_key) = issuer();
        let claims = claims(now_ms);
        let token = issue_route_capability(&private_key, &claims)?;
        let valid = verify_route_capability(
            &public_key,
            &token,
            &context(&claims, now_ms + 1_000),
            &BTreeSet::new(),
        )?;

        assert!(
            verify_route_capability(
                &public_key,
                &token,
                &context(&claims, claims.expires_at_ms + 1_000),
                &BTreeSet::new(),
            )
            .is_err()
        );
        assert!(
            verify_route_capability(
                &public_key,
                &token,
                &context(&claims, now_ms + 1_000),
                &BTreeSet::from([valid.root_revocation_id]),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn issuer_rejects_unbounded_or_malformed_claims() {
        let (private_key, _) = issuer();
        let mut invalid = claims(1_800_000_000_000);
        invalid.route_plan_digest = "not-a-digest".to_owned();
        assert!(issue_route_capability(&private_key, &invalid).is_err());

        let mut unbounded = claims(1_800_000_000_000);
        unbounded.expires_at_ms = unbounded.issued_at_ms + MAX_CAPABILITY_LIFETIME_MS + 1;
        assert!(issue_route_capability(&private_key, &unbounded).is_err());
    }

    #[test]
    fn public_key_derivation_matches_issuer() -> Result<()> {
        let (private_key, public_key) = issuer();
        assert_eq!(capability_public_key(&private_key)?, public_key);
        Ok(())
    }

    #[test]
    fn authenticated_root_revocation_id_matches_verification() -> Result<()> {
        let now_ms = 1_800_000_000_000;
        let (private_key, public_key) = issuer();
        let claims = claims(now_ms);
        let token = issue_route_capability(&private_key, &claims)?;
        let verified = verify_route_capability(
            &public_key,
            &token,
            &context(&claims, now_ms + 1_000),
            &BTreeSet::new(),
        )?;
        assert_eq!(
            route_capability_root_revocation_id(&public_key, &token)?,
            verified.root_revocation_id
        );
        Ok(())
    }
}
