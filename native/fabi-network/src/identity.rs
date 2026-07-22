use std::{
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    path::Path,
};

use anyhow::{Context, Result, bail, ensure};
use data_encoding::HEXLOWER;
use iroh::SecretKey;

const ENCODED_KEY_LEN: usize = 64;

/// Load a stable endpoint identity, creating it with owner-only permissions.
///
/// A malformed or partially-written key is never silently replaced: changing
/// identity would invalidate scheduler membership and peer allow-lists.
///
/// # Errors
///
/// Returns an error when the key cannot be created/read or its on-disk encoding
/// is malformed.
pub fn load_or_create(path: &Path) -> Result<SecretKey> {
    if path.exists() {
        return load(path);
    }

    let parent = path
        .parent()
        .context("identity path must have a parent directory")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create identity directory {}", parent.display()))?;

    let key = SecretKey::generate();
    let encoded = HEXLOWER.encode(&key.to_bytes());

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    match options.open(path) {
        Ok(mut file) => {
            file.write_all(encoded.as_bytes())
                .with_context(|| format!("failed to write identity {}", path.display()))?;
            file.write_all(b"\n")
                .with_context(|| format!("failed to finish identity {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("failed to sync identity {}", path.display()))?;
            Ok(key)
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => load(path),
        Err(error) => {
            Err(error).with_context(|| format!("failed to create identity {}", path.display()))
        }
    }
}

fn load(path: &Path) -> Result<SecretKey> {
    let mut encoded = String::new();
    File::open(path)
        .with_context(|| format!("failed to open identity {}", path.display()))?
        .read_to_string(&mut encoded)
        .with_context(|| format!("failed to read identity {}", path.display()))?;
    let encoded = encoded.trim();
    ensure!(
        encoded.len() == ENCODED_KEY_LEN,
        "identity {} is malformed: expected {ENCODED_KEY_LEN} hexadecimal characters",
        path.display()
    );
    let decoded = HEXLOWER
        .decode(encoded.as_bytes())
        .with_context(|| format!("identity {} is not lowercase hexadecimal", path.display()))?;
    let bytes: [u8; 32] = decoded.try_into().map_err(|_| {
        anyhow::anyhow!(
            "identity {} decoded to an invalid key length",
            path.display()
        )
    })?;
    if bytes.iter().all(|byte| *byte == 0) {
        bail!(
            "identity {} contains an invalid all-zero key",
            path.display()
        );
    }
    Ok(SecretKey::from_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_stable() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("worker.key");
        let first = load_or_create(&path).expect("create key");
        let second = load_or_create(&path).expect("load key");
        assert_eq!(first.public(), second.public());
    }

    #[test]
    fn malformed_identity_fails_closed() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("worker.key");
        fs::write(&path, "broken").expect("write malformed key");
        let error = load_or_create(&path).expect_err("malformed key must fail");
        assert!(error.to_string().contains("malformed"));
    }
}
