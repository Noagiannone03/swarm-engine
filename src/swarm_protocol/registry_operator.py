"""Operator tooling for the staging Fabi model registry.

Private TUF keys are deliberately kept outside the published repository.  They are serialized as
encrypted PKCS#8 PEM with pyca/cryptography and are only converted to securesystemslib signers in
memory.  This module is staging-oriented: production root ceremonies should use offline threshold
keys or a KMS/HSM signer behind the same :class:`RegistryRoleSigners` interface.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import secrets
import stat
import sys
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

from swarm_protocol.model_manifest import build_hub_model_bundle
from swarm_protocol.registry import (
    ModelRegistryBundle,
    RegistryRoleSigners,
    RouteAuthorityKeyset,
    TrustedModelRegistry,
    TufRegistryPublisher,
)

_KEY_ROLES = ("root", "targets", "snapshot", "timestamp")
_MAX_SECRET_BYTES = 4096
_MAX_BUNDLE_BYTES = 32 * 1024 * 1024
_MAX_ROUTE_AUTHORITY_BYTES = 1024 * 1024


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
