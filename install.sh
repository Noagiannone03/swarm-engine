#!/bin/bash
# Install Parallax from source and build the vLLM Rust frontend binary.
# Usage:
#   ./install.sh [--extras EXTRAS] [--python PYTHON_VERSION]
#   PARALLAX_VENV_DIR=/path/to/venv ./install.sh --frontend-only
#
# By default, installs mac extras on macOS and gpu extras on Linux, then
# builds vllm-rs in release mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXTRAS="${PARALLAX_EXTRAS:-}"
PYTHON_VERSION="${PARALLAX_PYTHON_VERSION:-3.12}"
VENV_DIR="${PARALLAX_VENV_DIR:-$SCRIPT_DIR/.venv}"
VLLM_REF="${VLLM_REF:-ee0da84ab9e04ac7610e28580af62c365e898389}"
VLLM_MINIJINJA_VERSION="${VLLM_MINIJINJA_VERSION-2.20.0}"
VLLM_FABI_PATCH="$SCRIPT_DIR/patches/vllm-v0.24.0-fabi-chat-replay.patch"
VLLM_PORTABLE_FRONTEND_PATCH="$SCRIPT_DIR/patches/vllm-v0.24.0-portable-frontend.patch"
FRONTEND_ONLY=false

show_help() {
    cat <<'EOF'
Install Parallax from source and build the vLLM Rust frontend binary.

Usage:
  ./install.sh [--extras EXTRAS] [--python PYTHON_VERSION]
  PARALLAX_VENV_DIR=/path/to/venv ./install.sh --frontend-only

Options:
  --extras EXTRAS         Python extras to install, for example "mac", "gpu",
                          or "mac,dev". Defaults to mac on macOS and gpu on Linux.
  --python PYTHON_VERSION Python version for uv venv. Defaults to 3.12.
  --frontend-only         Build vllm-rs into an existing virtualenv.
                          Does not create a venv or install Python packages.

Environment:
  PARALLAX_EXTRAS         Same as --extras.
  PARALLAX_PYTHON_VERSION Same as --python.
  PARALLAX_VENV_DIR       Virtualenv to create/use. Defaults to ./.venv.
  VLLM_REF                vLLM git branch, tag, or full commit hash to clone.
                          Defaults to the immutable v0.24.0 commit qualified by Fabi.
  VLLM_MINIJINJA_VERSION  MiniJinja/minijinja-contrib version to use when
                          building vllm-rs. Defaults to 2.20.0.
EOF
}

usage_error() {
    echo "$1" >&2
    show_help >&2
    exit 2
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --extras)
                [[ $# -ge 2 ]] || usage_error "--extras requires a value."
                EXTRAS="$2"
                shift 2
                ;;
            --extras=*)
                EXTRAS="${1#*=}"
                shift
                ;;
            --python)
                [[ $# -ge 2 ]] || usage_error "--python requires a value."
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --python=*)
                PYTHON_VERSION="${1#*=}"
                shift
                ;;
            --frontend-only)
                FRONTEND_ONLY=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                usage_error "Unknown argument: $1"
                ;;
        esac
    done
}

normalize_config() {
    EXTRAS="${EXTRAS//[[:space:]]/}"
    if [[ -z "$EXTRAS" ]]; then
        case "$(uname -s)" in
            Darwin) EXTRAS="mac" ;;
            Linux) EXTRAS="gpu" ;;
            *) EXTRAS="" ;;
        esac
    fi
}

run_with_sudo_if_needed() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo &>/dev/null; then
        sudo "$@"
    else
        echo "Need root privileges to run: $*" >&2
        echo "Install the dependency manually, then rerun this script." >&2
        exit 1
    fi
}

ensure_uv() {
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if command -v uv &>/dev/null; then
        return
    fi

    if ! command -v curl &>/dev/null; then
        echo "curl not found; install curl and rerun this script." >&2
        exit 1
    fi

    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "uv installation completed, but uv is still not on PATH." >&2
        exit 1
    fi
}

ensure_venv() {
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        echo "Using existing virtual environment: $VENV_DIR"
        return
    fi

    echo "Creating virtual environment: $VENV_DIR (Python $PYTHON_VERSION)"
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        echo "Virtual environment was created, but $VENV_DIR/bin/python is missing." >&2
        exit 1
    fi
}

install_parallax_python() {
    local venv_python="$VENV_DIR/bin/python"
    local package_spec="."

    if [[ -n "$EXTRAS" ]]; then
        package_spec=".[${EXTRAS}]"
    fi

    echo "Installing Parallax Python package: $package_spec"
    UV_PRERELEASE=allow uv pip install --python "$venv_python" -e "$package_spec"
}

resolve_venv_bin_dir() {
    "$(resolve_venv_python)" - <<'PY'
import sysconfig

print(sysconfig.get_path("scripts"))
PY
}

resolve_venv_python() {
    if [[ -x "$VENV_DIR/bin/python" ]]; then
        printf '%s\n' "$VENV_DIR/bin/python"
        return
    fi
    if [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
        printf '%s\n' "$VENV_DIR/Scripts/python.exe"
        return
    fi
    echo "Existing virtualenv Python is missing under $VENV_DIR." >&2
    return 1
}

ensure_git() {
    if ! command -v git &>/dev/null; then
        echo "git not found; install git and rerun this script." >&2
        exit 1
    fi
}

ensure_protoc() {
    if command -v protoc &>/dev/null; then
        return
    fi

    echo "protoc not found, installing protobuf compiler..."

    if command -v brew &>/dev/null; then
        brew install protobuf
    elif command -v apt-get &>/dev/null; then
        run_with_sudo_if_needed apt-get update
        run_with_sudo_if_needed apt-get install -y protobuf-compiler
    elif command -v dnf &>/dev/null; then
        run_with_sudo_if_needed dnf install -y protobuf-compiler
    elif command -v yum &>/dev/null; then
        run_with_sudo_if_needed yum install -y protobuf-compiler
    elif command -v apk &>/dev/null; then
        run_with_sudo_if_needed apk add --no-cache protobuf
    elif command -v choco &>/dev/null; then
        choco install protoc -y
    else
        echo "Unable to install protoc automatically." >&2
        echo "Install protobuf compiler manually, then rerun this script." >&2
        exit 1
    fi

    if ! command -v protoc &>/dev/null; then
        echo "protoc installation completed, but protoc is still not on PATH." >&2
        exit 1
    fi
}

ensure_rust_toolchain() {
    local toolchain="$1"

    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v rustup &>/dev/null; then
        if ! command -v curl &>/dev/null; then
            echo "curl not found; install curl and rerun this script." >&2
            exit 1
        fi

        echo "rustup not found, installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
            sh -s -- -y --default-toolchain none
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi

    if ! rustup run "$toolchain" rustc --version &>/dev/null; then
        echo "Installing Rust toolchain: $toolchain"
        rustup toolchain install "$toolchain"
    fi
}

build_vllm_rust_frontend() {
    local vllm_clone_root
    local rust_dir
    local parallax_scripts_dir
    local target_path
    local target_version_path
    local target_version
    local existing_version
    local toolchain
    local executable_suffix
    local built_binary

    parallax_scripts_dir="$(resolve_venv_bin_dir)"
    executable_suffix="$("$(resolve_venv_python)" - <<'PY'
import sysconfig

print(sysconfig.get_config_var("EXE") or "")
PY
)"
    target_path="$parallax_scripts_dir/vllm-rs$executable_suffix"
    target_version_path="$target_path.version"
    target_version="$(vllm_rust_frontend_version)"

    if [[ -f "$target_path" ]]; then
        existing_version=""
        if [[ -f "$target_version_path" ]]; then
            existing_version="$(<"$target_version_path")"
        fi
        if [[ "$existing_version" != "$target_version" ]]; then
            echo "Existing vllm-rs version (${existing_version:-unknown}) does not match $target_version, rebuilding."
            rm -f "$target_path" "$target_version_path"
        else
            chmod +x "$target_path"
            printf '%s\n' "$target_version" > "$target_version_path"
            echo "vllm-rs already exists at $target_path, skipping Rust build."
            return
        fi
    fi

    CLONE_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/parallax-vllm-rs.XXXXXX")"
    vllm_clone_root="$CLONE_PARENT/vllm"
    trap cleanup_clone EXIT

    clone_vllm_ref "$vllm_clone_root"

    rust_dir="$vllm_clone_root/rust"

    if [[ ! -f "$rust_dir/Cargo.toml" || ! -f "$vllm_clone_root/rust-toolchain.toml" ]]; then
        echo "Cloned repository does not contain the expected vLLM Rust frontend sources." >&2
        exit 1
    fi

    toolchain=$(grep -E '^[[:space:]]*channel' "$vllm_clone_root/rust-toolchain.toml" |
        sed 's/.*= *"\(.*\)"/\1/')

    if [[ -z "$toolchain" ]]; then
        echo "Unable to read Rust toolchain from $vllm_clone_root/rust-toolchain.toml" >&2
        exit 1
    fi

    ensure_rust_toolchain "$toolchain"
    ensure_protoc
    apply_vllm_fabi_patch "$vllm_clone_root"
    update_vllm_minijinja "$rust_dir" "$toolchain"

    # pcre2-sys préfère la bibliothèque dynamique trouvée sur la machine de
    # build. Sur macOS/Homebrew cela grave /opt/homebrew/... dans vllm-rs et le
    # binaire ne démarre plus sur une machine sans ce keg. Le crate documente
    # PCRE2_SYS_STATIC=1 comme contrat officiel pour builder PCRE2 depuis les
    # sources et le lier statiquement.
    PCRE2_SYS_STATIC=1 cargo +"$toolchain" build --release \
        --manifest-path "$rust_dir/Cargo.toml" \
        --bin vllm-rs \
        --features native-tls-vendored

    built_binary="$rust_dir/target/release/vllm-rs$executable_suffix"
    bash "$SCRIPT_DIR/scripts/check-vllm-rs-portability.sh" \
        "$built_binary"

    mkdir -p "$(dirname "$target_path")"
    cp "$built_binary" "$target_path"
    chmod +x "$target_path"
    printf '%s\n' "$target_version" > "$target_version_path"
    echo "Installed vllm-rs to $target_path"
    cleanup_clone
    trap - EXIT
}

vllm_rust_frontend_version() {
    local replay_patch_hash
    local portable_patch_hash
    local venv_python

    venv_python="$(resolve_venv_python)"
    replay_patch_hash="$("$venv_python" - "$VLLM_FABI_PATCH" <<'PY'
import hashlib
from pathlib import Path
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
    portable_patch_hash="$("$venv_python" - "$VLLM_PORTABLE_FRONTEND_PATCH" <<'PY'
import hashlib
from pathlib import Path
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
    if [[ -n "$VLLM_MINIJINJA_VERSION" ]]; then
        printf '%s+minijinja-%s+fabi-%s+portable-%s\n' \
            "$VLLM_REF" "$VLLM_MINIJINJA_VERSION" \
            "$replay_patch_hash" "$portable_patch_hash"
    else
        printf '%s+fabi-%s+portable-%s\n' \
            "$VLLM_REF" "$replay_patch_hash" "$portable_patch_hash"
    fi
}

apply_vllm_fabi_patch() {
    local clone_root="$1"

    if [[ ! -f "$VLLM_FABI_PATCH" ]]; then
        echo "Required vLLM Fabi patch is missing: $VLLM_FABI_PATCH" >&2
        exit 1
    fi

    echo "Applying Fabi token-exact chat replay patch"
    if ! git -C "$clone_root" apply --unidiff-zero --check "$VLLM_FABI_PATCH"; then
        echo "The Fabi replay patch is incompatible with vLLM ref $VLLM_REF." >&2
        echo "Qualify and update the patch before building this frontend." >&2
        exit 1
    fi
    git -C "$clone_root" apply --unidiff-zero "$VLLM_FABI_PATCH"

    if [[ ! -f "$VLLM_PORTABLE_FRONTEND_PATCH" ]]; then
        echo "Required portable frontend patch is missing: $VLLM_PORTABLE_FRONTEND_PATCH" >&2
        exit 1
    fi

    echo "Applying Fabi portable frontend listener patch"
    if ! git -C "$clone_root" apply --check "$VLLM_PORTABLE_FRONTEND_PATCH"; then
        echo "The portable frontend patch is incompatible with vLLM ref $VLLM_REF." >&2
        echo "Qualify and update the patch before building this frontend." >&2
        exit 1
    fi
    git -C "$clone_root" apply "$VLLM_PORTABLE_FRONTEND_PATCH"
}

update_vllm_minijinja() {
    local rust_dir="$1"
    local toolchain="$2"

    if [[ -z "$VLLM_MINIJINJA_VERSION" ]]; then
        return
    fi

    echo "Updating vLLM Rust MiniJinja dependencies to $VLLM_MINIJINJA_VERSION"
    cargo +"$toolchain" update \
        --manifest-path "$rust_dir/Cargo.toml" \
        -p minijinja \
        --precise "$VLLM_MINIJINJA_VERSION"
    cargo +"$toolchain" update \
        --manifest-path "$rust_dir/Cargo.toml" \
        -p minijinja-contrib \
        --precise "$VLLM_MINIJINJA_VERSION"
}

clone_vllm_ref() {
    local clone_root="$1"
    local repo_url="https://github.com/vllm-project/vllm.git"
    local fetched=false

    echo "Cloning vLLM from $repo_url (ref: $VLLM_REF)"
    git init "$clone_root"
    git -C "$clone_root" remote add origin "$repo_url"

    if git -C "$clone_root" fetch --depth 1 origin "refs/heads/$VLLM_REF" 2>/dev/null; then
        fetched=true
    elif git -C "$clone_root" fetch --depth 1 origin "refs/tags/$VLLM_REF" 2>/dev/null; then
        fetched=true
    elif git -C "$clone_root" fetch --depth 1 origin "$VLLM_REF"; then
        fetched=true
    fi

    if [[ "$fetched" != true ]]; then
        echo "Unable to fetch vLLM ref: $VLLM_REF" >&2
        echo "Set VLLM_REF to a valid vLLM branch, tag, or full commit hash." >&2
        exit 1
    fi

    git -C "$clone_root" checkout --detach FETCH_HEAD
    echo "Checked out vLLM commit $(git -C "$clone_root" rev-parse HEAD)"
}

cleanup_clone() {
    if [[ -n "${CLONE_PARENT:-}" ]]; then
        rm -rf "$CLONE_PARENT"
    fi
}

main() {
    parse_args "$@"
    normalize_config

    cd "$SCRIPT_DIR"

    if [[ "$FRONTEND_ONLY" == true ]]; then
        if ! resolve_venv_python >/dev/null; then
            echo "Existing virtualenv is required for --frontend-only: $VENV_DIR" >&2
            exit 1
        fi
        ensure_git
        build_vllm_rust_frontend
        echo "vllm-rs frontend build complete."
        return
    fi

    ensure_uv
    ensure_venv
    install_parallax_python
    ensure_git
    build_vllm_rust_frontend

    echo "Install complete."
    echo "Activate the environment with: source .venv/bin/activate"
}

main "$@"
