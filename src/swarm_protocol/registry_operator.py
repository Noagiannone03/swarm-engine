"""Operator tooling for the staging Fabi model registry.

Private TUF keys are deliberately kept outside the published repository.  They are serialized as
encrypted PKCS#8 PEM with pyca/cryptography and are only converted to securesystemslib signers in
memory.  This module is staging-oriented: production root ceremonies should use offline threshold
keys or a KMS/HSM signer behind the same :class:`RegistryRoleSigners` interface.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    PrivateFormat,
    load_pem_private_key,
)
from securesystemslib.signer import CryptoSigner

from fabi_network.capability import capability_public_key
from swarm_protocol.model_manifest import build_hub_model_bundle
from swarm_protocol.registry import (
    ModelRegistryBundle,
    RouteAuthorityKey,
    RegistryRoleSigners,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    TufRegistryPublisher,
    TufTimestampRefresher,
)

_KEY_ROLES = ("root", "targets", "snapshot", "timestamp")
_MAX_SECRET_BYTES = 4096
_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
_MAX_ROUTE_AUTHORITY_BYTES = 1024 * 1024
_DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS = 90
_DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS = 300


def _assert_private_path(path: Path, *, description: str) -> None:
    """Reject group/world-readable operator secrets on Unix."""

    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"{description} must not be accessible by group or others: {path}")


def read_passphrase_file(path: Path) -> bytes:
    """Read a bounded passphrase from an owner-only file without logging it."""

    _assert_private_path(path, description="registry passphrase file")
    with path.open("rb") as source:
        payload = source.read(_MAX_SECRET_BYTES + 1)
    if len(payload) > _MAX_SECRET_BYTES:
        raise ValueError(f"registry passphrase exceeds {_MAX_SECRET_BYTES} bytes")
    # Supporting a conventional one-line secret file is useful for operator automation.  Remove
    # only its line terminator: spaces and all other bytes remain part of the passphrase.
    passphrase = payload.rstrip(b"\r\n")
    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    return passphrase


def generate_passphrase_file(path: Path) -> None:
    """Create a high-entropy staging passphrase file without printing its contents."""

    # token_urlsafe(32) contains 256 bits of CSPRNG entropy before encoding and avoids embedded
    # newlines, making the resulting owner-only file suitable for unattended staging publication.
    _atomic_create_private(path, secrets.token_urlsafe(32).encode("ascii"))


def prompt_passphrase(*, confirm: bool) -> bytes:
    """Read a passphrase from the terminal, optionally confirming a new secret."""

    first = getpass.getpass("Registry key passphrase: ").encode("utf-8")
    if not first:
        raise ValueError("registry passphrase must not be empty")
    if confirm:
        second = getpass.getpass("Confirm registry key passphrase: ").encode("utf-8")
        if first != second:
            raise ValueError("registry passphrase confirmation does not match")
    return first


def _atomic_create_private(path: Path, payload: bytes) -> None:
    """Create one private file without ever replacing existing key material."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_create_public(path: Path, payload: bytes) -> None:
    """Create one public operator artifact without replacing an existing generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def generate_route_authority(
    private_key_output: Path,
    keyset_output: Path,
    *,
    generation: int = 1,
    valid_for_days: int = _DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS,
    clock_skew_seconds: int = _DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS,
    now_ms: int | None = None,
) -> RouteAuthorityKeyset:
    """Generate an owner-only Biscuit seed and its public TUF keyset.

    The private seed is never written to stdout and is deliberately separate
    from the public keyset that the targets role signs.
    """

    if private_key_output == keyset_output:
        raise ValueError("route authority private key and keyset paths must be different")
    if private_key_output.exists():
        raise FileExistsError(f"refusing to overwrite route authority key: {private_key_output}")
    if keyset_output.exists():
        raise FileExistsError(f"refusing to overwrite route authority keyset: {keyset_output}")
    if generation <= 0:
        raise ValueError("route authority generation must be positive")
    if not 1 <= valid_for_days <= 3650:
        raise ValueError("route authority validity must be between 1 and 3650 days")
    if not 0 <= clock_skew_seconds <= 3600:
        raise ValueError("route authority clock skew must be between 0 and 3600 seconds")

    current_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    if current_ms < 0:
        raise ValueError("route authority publication time must not be negative")
    valid_for_ms = valid_for_days * 24 * 60 * 60 * 1000
    skew_ms = clock_skew_seconds * 1000
    issued_at_ms = max(0, current_ms - skew_ms)
    expires_at_ms = current_ms + valid_for_ms
    private_key = secrets.token_hex(32)
    public_key = capability_public_key(private_key)
    authority = RouteAuthorityKey(
        key_id=hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
        public_key=public_key,
        not_before_ms=issued_at_ms,
        not_after_ms=expires_at_ms,
    )
    keyset = RouteAuthorityKeyset(
        generation=generation,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
        keys=(authority,),
    )

    created_private = False
    try:
        _atomic_create_private(private_key_output, private_key.encode("ascii") + b"\n")
        created_private = True
        _atomic_create_public(keyset_output, keyset.canonical_bytes() + b"\n")
    except BaseException:
        if created_private:
            try:
                private_key_output.unlink()
            except FileNotFoundError:
                pass
        raise
    return keyset


def generate_staging_keys(key_dir: Path, passphrase: bytes) -> RegistryRoleSigners:
    """Generate one encrypted Ed25519 key per TUF role and return in-memory signers."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    expected_paths = tuple(key_dir / f"{role}.pem" for role in _KEY_ROLES)
    existing = [path for path in expected_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite registry key: {existing[0]}")

    signers: dict[str, CryptoSigner] = {}
    created: list[Path] = []
    try:
        for role, path in zip(_KEY_ROLES, expected_paths, strict=True):
            private_key = Ed25519PrivateKey.generate()
            pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=BestAvailableEncryption(passphrase),
            )
            _atomic_create_private(path, pem)
            created.append(path)
            signers[role] = CryptoSigner(private_key)
    except BaseException:
        # A partially-created authority is unsafe and confusing.  Only remove files created by
        # this call; pre-existing paths were rejected before the first write.
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    return RegistryRoleSigners(
        root=(signers["root"],),
        targets=(signers["targets"],),
        snapshot=(signers["snapshot"],),
        timestamp=(signers["timestamp"],),
    )


def load_staging_keys(key_dir: Path, passphrase: bytes) -> RegistryRoleSigners:
    """Load the four encrypted staging keys and enforce their type and permissions."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    signers: dict[str, CryptoSigner] = {}
    for role in _KEY_ROLES:
        path = key_dir / f"{role}.pem"
        _assert_private_path(path, description=f"registry {role} private key")
        private_key = load_pem_private_key(path.read_bytes(), password=passphrase)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError(f"registry {role} key is not Ed25519")
        signers[role] = CryptoSigner(private_key)
    return RegistryRoleSigners(
        root=(signers["root"],),
        targets=(signers["targets"],),
        snapshot=(signers["snapshot"],),
        timestamp=(signers["timestamp"],),
    )


def load_staging_timestamp_signers(
    key_dir: Path,
    passphrase: bytes,
) -> tuple[CryptoSigner, ...]:
    """Load only the encrypted online timestamp signer."""

    if not passphrase:
        raise ValueError("registry passphrase must not be empty")
    path = key_dir / "timestamp.pem"
    _assert_private_path(path, description="registry timestamp private key")
    private_key = load_pem_private_key(path.read_bytes(), password=passphrase)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("registry timestamp key is not Ed25519")
    return (CryptoSigner(private_key),)


def load_bundles(paths: Sequence[Path]) -> tuple[ModelRegistryBundle, ...]:
    """Load complete immutable registry bundles from bounded JSON files."""

    if not paths:
        raise ValueError("at least one model bundle is required")
    bundles = []
    for path in paths:
        if path.stat().st_size > _MAX_BUNDLE_BYTES:
            raise ValueError(f"model bundle exceeds {_MAX_BUNDLE_BYTES} bytes: {path}")
        bundles.append(ModelRegistryBundle.model_validate_json(path.read_bytes()))
    return tuple(bundles)


def load_route_authorities(path: Path | None) -> RouteAuthorityKeyset | None:
    """Load a bounded public keyset; the TUF targets role signs it on publish."""

    if path is None:
        return None
    if path.stat().st_size > _MAX_ROUTE_AUTHORITY_BYTES:
        raise ValueError("route authority keyset exceeds 1 MiB")
    return RouteAuthorityKeyset.model_validate_json(path.read_bytes())


def _atomic_write_public(path: Path, payload: bytes) -> None:
    """Write a non-secret operator artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_hub_bundle_file(
    output: Path,
    *,
    model_id: str,
    revision: str | None,
    quantization: str,
    dtype: str,
    token: bool | str | None = None,
) -> ModelRegistryBundle:
    """Resolve Hub metadata and persist one canonical registry bundle."""

    resolved = build_hub_model_bundle(
        model_id,
        revision=revision,
        quantization=quantization,
        dtype=dtype,
        token=token,
        include_weight_profile=True,
        include_selective_weight_index=True,
    )
    bundle = ModelRegistryBundle(
        manifest=resolved.manifest,
        artifact_index=resolved.artifact_index,
    )
    _atomic_write_public(output, bundle.canonical_bytes() + b"\n")
    return bundle


def initialize_staging_registry(
    repository_dir: Path,
    key_dir: Path,
    bundle_paths: Sequence[Path],
    passphrase: bytes,
    *,
    bootstrap_root_output: Path,
    route_authority_path: Path | None = None,
) -> tuple[ModelRegistryBundle, ...]:
    """Create a fresh 1-of-1 staging authority and its first signed snapshot."""

    if repository_dir.exists() and any(repository_dir.iterdir()):
        raise FileExistsError("refusing to initialize a non-empty registry repository")
    if bootstrap_root_output.exists():
        raise FileExistsError(f"refusing to overwrite bootstrap root: {bootstrap_root_output}")
    bundles = load_bundles(bundle_paths)
    route_authorities = load_route_authorities(route_authority_path)
    signers = generate_staging_keys(key_dir, passphrase)
    try:
        root = TufRegistryPublisher(repository_dir, signers).initialize(
            bundles,
            route_authorities=route_authorities,
        )
        _atomic_write_public(bootstrap_root_output, root)
    except BaseException:
        # Key deletion is intentionally not automatic here: once an authority has been created,
        # retaining its private keys is safer than silently replacing it on a later retry.
        raise
    return bundles


def publish_staging_registry(
    repository_dir: Path,
    key_dir: Path,
    bundle_paths: Sequence[Path],
    passphrase: bytes,
    *,
    route_authority_path: Path | None = None,
) -> tuple[int, tuple[ModelRegistryBundle, ...]]:
    """Publish a complete new staging snapshot with already-created keys."""

    bundles = load_bundles(bundle_paths)
    route_authorities = load_route_authorities(route_authority_path)
    version = TufRegistryPublisher(
        repository_dir,
        load_staging_keys(key_dir, passphrase),
    ).publish(bundles, route_authorities=route_authorities)
    return version, bundles


def _passphrase(args: argparse.Namespace, *, confirm: bool) -> bytes:
    if args.passphrase_file is not None:
        return read_passphrase_file(args.passphrase_file)
    if not sys.stdin.isatty():
        raise ValueError("non-interactive use requires --passphrase-file")
    return prompt_passphrase(confirm=confirm)


def _bundle_summary(bundle: ModelRegistryBundle) -> dict[str, object]:
    return {
        "model_id": bundle.manifest.model_id,
        "immutable_revision": bundle.manifest.immutable_revision,
        "model_swarm_id": bundle.model_swarm_id,
        "num_layers": bundle.manifest.num_layers,
        "model_max_context_tokens": bundle.manifest.model_max_context_tokens,
        "context_classes": list(bundle.manifest.context_classes),
        "artifacts": len(bundle.artifact_index.artifacts),
        "tensors": len(bundle.artifact_index.tensors),
        "signed_tensor_bytes": sum(tensor.length for tensor in bundle.artifact_index.tensors),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabi-swarm-registry",
        description="Build and publish the signed Fabi staging model registry",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    passphrase = commands.add_parser("generate-passphrase")
    passphrase.add_argument("--output", type=Path, required=True)

    route_authority = commands.add_parser("generate-route-authority")
    route_authority.add_argument("--private-key-output", type=Path, required=True)
    route_authority.add_argument("--keyset-output", type=Path, required=True)
    route_authority.add_argument("--generation", type=int, default=1)
    route_authority.add_argument(
        "--valid-for-days",
        type=int,
        default=_DEFAULT_ROUTE_AUTHORITY_VALIDITY_DAYS,
    )
    route_authority.add_argument(
        "--clock-skew-seconds",
        type=int,
        default=_DEFAULT_ROUTE_AUTHORITY_CLOCK_SKEW_SECONDS,
    )

    build = commands.add_parser("build-hub-bundle")
    build.add_argument("--model-id", required=True)
    build.add_argument("--revision")
    build.add_argument("--quantization", required=True)
    build.add_argument("--dtype", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--use-hf-token",
        action="store_true",
        help="use the token from the normal Hugging Face credential store",
    )

    initialize = commands.add_parser("init-staging")
    initialize.add_argument("--repository-dir", type=Path, required=True)
    initialize.add_argument("--key-dir", type=Path, required=True)
    initialize.add_argument("--bundle", type=Path, action="append", required=True)
    initialize.add_argument("--bootstrap-root-output", type=Path, required=True)
    initialize.add_argument("--route-authorities", type=Path)
    initialize.add_argument("--passphrase-file", type=Path)

    publish = commands.add_parser("publish")
    publish.add_argument("--repository-dir", type=Path, required=True)
    publish.add_argument("--key-dir", type=Path, required=True)
    publish.add_argument("--bundle", type=Path, action="append", required=True)
    publish.add_argument("--route-authorities", type=Path)
    publish.add_argument("--passphrase-file", type=Path)

    refresh_timestamp = commands.add_parser("refresh-timestamp")
    refresh_timestamp.add_argument("--repository-dir", type=Path, required=True)
    refresh_timestamp.add_argument("--key-dir", type=Path, required=True)
    refresh_timestamp.add_argument("--passphrase-file", type=Path)

    verify = commands.add_parser("verify-remote")
    verify.add_argument("--bootstrap-root", type=Path, required=True)
    verify.add_argument("--metadata-url", required=True)
    verify.add_argument("--targets-url", required=True)
    verify.add_argument("--state-dir", type=Path, required=True)
    identity = verify.add_mutually_exclusive_group(required=True)
    identity.add_argument("--model-swarm-id")
    identity.add_argument("--model-id")
    verify.add_argument("--revision")
    verify.add_argument("--quantization")
    verify.add_argument("--dtype")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the operator CLI and emit only non-secret JSON summaries."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    if args.command == "generate-passphrase":
        generate_passphrase_file(args.output)
        result = {"status": "generated", "output": str(args.output)}
    elif args.command == "generate-route-authority":
        keyset = generate_route_authority(
            args.private_key_output,
            args.keyset_output,
            generation=args.generation,
            valid_for_days=args.valid_for_days,
            clock_skew_seconds=args.clock_skew_seconds,
        )
        result = {
            "status": "generated",
            "generation": keyset.generation,
            "key_id": keyset.keys[0].key_id,
            "expires_at_ms": keyset.expires_at_ms,
            "private_key_output": str(args.private_key_output),
            "keyset_output": str(args.keyset_output),
        }
    elif args.command == "build-hub-bundle":
        bundle = build_hub_bundle_file(
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            token=True if args.use_hf_token else None,
        )
        result = {"status": "built", **_bundle_summary(bundle), "output": str(args.output)}
    elif args.command == "init-staging":
        bundles = initialize_staging_registry(
            args.repository_dir,
            args.key_dir,
            args.bundle,
            _passphrase(args, confirm=True),
            bootstrap_root_output=args.bootstrap_root_output,
            route_authority_path=args.route_authorities,
        )
        result = {
            "status": "initialized_staging_1_of_1",
            "models": [_bundle_summary(bundle) for bundle in bundles],
            "repository": str(args.repository_dir),
            "bootstrap_root": str(args.bootstrap_root_output),
        }
    elif args.command == "publish":
        version, bundles = publish_staging_registry(
            args.repository_dir,
            args.key_dir,
            args.bundle,
            _passphrase(args, confirm=False),
            route_authority_path=args.route_authorities,
        )
        result = {
            "status": "published",
            "version": version,
            "models": [_bundle_summary(bundle) for bundle in bundles],
        }
    elif args.command == "refresh-timestamp":
        version = TufTimestampRefresher(
            args.repository_dir,
            load_staging_timestamp_signers(
                args.key_dir,
                _passphrase(args, confirm=False),
            ),
        ).refresh()
        result = {
            "status": "timestamp_refreshed",
            "version": version,
            "repository": str(args.repository_dir),
        }
    else:
        registry = TrustedModelRegistry(
            args.state_dir,
            metadata_base_url=args.metadata_url,
            target_base_url=args.targets_url,
            bootstrap_root=args.bootstrap_root.read_bytes(),
        )
        if args.model_swarm_id is not None:
            bundle = registry.fetch(args.model_swarm_id)
        else:
            bundle = registry.resolve(
                args.model_id,
                immutable_revision=args.revision,
                quantization=args.quantization,
                dtype=args.dtype,
            )
        result = {"status": "verified", **_bundle_summary(bundle)}

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
