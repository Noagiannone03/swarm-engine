use std::{fs, path::Path, time::Duration};

use anyhow::{Context, Result, bail};
use iroh::{Endpoint, RelayMap, RelayMode, RelayUrl, SecretKey, Watcher, endpoint::presets};
use n0_future::StreamExt;

use crate::ALPN;

const ONLINE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct EndpointConfig {
    pub relay_url: RelayUrl,
    pub relay_token: Option<String>,
    pub force_relay: bool,
}

impl EndpointConfig {
    /// Read and trim a relay bearer token.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or contains no token.
    pub fn token_from_file(path: Option<&Path>) -> Result<Option<String>> {
        let Some(path) = path else {
            return Ok(None);
        };
        let token = fs::read_to_string(path)
            .with_context(|| format!("failed to read relay token {}", path.display()))?;
        let token = token.trim();
        if token.is_empty() {
            bail!("relay token {} is empty", path.display());
        }
        Ok(Some(token.to_owned()))
    }
}

/// Bind an authenticated endpoint and wait until its relay is online.
///
/// # Errors
///
/// Returns an error when the endpoint cannot bind or the configured relay does
/// not become reachable before the startup deadline.
pub async fn bind(secret_key: SecretKey, config: &EndpointConfig) -> Result<Endpoint> {
    let mut relay_map = RelayMap::from(config.relay_url.clone());
    if let Some(token) = &config.relay_token {
        relay_map = relay_map.with_auth_token(token.clone());
    }

    let mut builder = Endpoint::builder(presets::N0)
        .secret_key(secret_key)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(RelayMode::Custom(relay_map));
    if config.force_relay {
        builder = builder.clear_ip_transports();
    }

    let endpoint = builder
        .bind()
        .await
        .context("failed to bind Iroh endpoint")?;
    if let Err(error) = wait_for_relay(&endpoint).await {
        endpoint.close().await;
        return Err(error);
    }
    Ok(endpoint)
}

async fn wait_for_relay(endpoint: &Endpoint) -> Result<()> {
    let mut statuses = endpoint.home_relay_status().stream();
    let deadline = tokio::time::sleep(ONLINE_TIMEOUT);
    tokio::pin!(deadline);
    let mut last_error = None;

    loop {
        tokio::select! {
            () = &mut deadline => {
                let suffix = last_error
                    .map(|error: String| format!("; last relay error: {error}"))
                    .unwrap_or_default();
                bail!("endpoint did not connect to the configured relay within 30 seconds{suffix}");
            }
            update = statuses.next() => {
                let Some(update) = update else {
                    bail!("relay status stream ended before the endpoint came online");
                };
                if update.iter().any(iroh::endpoint::RelayStatus::is_connected) {
                    return Ok(());
                }
                for error in update.iter().filter_map(iroh::endpoint::RelayStatus::last_error) {
                    let message = format!("{error:#}");
                    if message.contains("not authorized") {
                        bail!("relay authentication failed: {message}");
                    }
                    last_error = Some(message);
                }
            }
        }
    }
}
