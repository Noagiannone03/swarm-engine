use iroh::endpoint::Connection;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct PathSnapshot {
    pub selected: bool,
    pub kind: &'static str,
    pub remote: String,
    pub rtt_ms: f64,
}

#[must_use]
pub fn snapshot(connection: &Connection) -> Vec<PathSnapshot> {
    connection
        .paths()
        .iter()
        .map(|path| PathSnapshot {
            selected: path.is_selected(),
            kind: if path.is_ip() {
                "direct"
            } else if path.is_relay() {
                "relay"
            } else {
                "custom"
            },
            remote: path.remote_addr().to_string(),
            rtt_ms: path.rtt().as_secs_f64() * 1_000.0,
        })
        .collect()
}
