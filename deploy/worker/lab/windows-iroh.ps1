$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$state = Join-Path $env:LOCALAPPDATA "fabi"
$runtime = Join-Path $state "runtime"
$python = Join-Path $runtime "parallax-venv\Scripts\python.exe"
$sourcePointer = Join-Path $state "runtime-candidate-current.txt"
$schedulerPointer = Join-Path $state "scheduler-endpoint.txt"
$source = if ($env:FABI_PARALLAX_SOURCE) {
  $env:FABI_PARALLAX_SOURCE
} else {
  (Get-Content $sourcePointer -Raw).Trim()
}
$schedulerEndpoint = if ($env:FABI_SCHEDULER_ENDPOINT) {
  $env:FABI_SCHEDULER_ENDPOINT
} else {
  (Get-Content $schedulerPointer -Raw).Trim()
}
$accountTokenFile = if ($env:FABI_ACCOUNT_TOKEN_FILE) {
  $env:FABI_ACCOUNT_TOKEN_FILE
} else {
  Join-Path $HOME ".config\fabi\account-token"
}
$relayTokenFile = if ($env:FABI_RELAY_TOKEN_FILE) {
  $env:FABI_RELAY_TOKEN_FILE
} else {
  Join-Path $state "network\relay.env"
}

if (-not (Test-Path $python -PathType Leaf)) { throw "worker Python not found: $python" }
if (-not (Test-Path (Join-Path $source "pyproject.toml") -PathType Leaf)) { throw "invalid Parallax source: $source" }
if ($schedulerEndpoint -notmatch "^[0-9a-f]{64}$") { throw "invalid Iroh scheduler endpoint" }
if (-not (Test-Path $accountTokenFile -PathType Leaf)) { throw "account token file is not readable" }
if (-not (Test-Path $relayTokenFile -PathType Leaf)) { throw "relay token file is not readable" }

$networkState = Join-Path $state "network"
$processLogs = Join-Path $state "process-logs"
$outLog = Join-Path $state "worker-windows-iroh.out.log"
$errLog = Join-Path $state "worker-windows-iroh.err.log"
New-Item -ItemType Directory -Force -Path $networkState, $processLogs | Out-Null

$env:FABI_ACCOUNT_TOKEN = (Get-Content $accountTokenFile -Raw).Trim()
$env:FABI_NETWORK_TRANSPORT = "iroh"
$env:FABI_RELAY_URL = if ($env:FABI_RELAY_URL) { $env:FABI_RELAY_URL } else { "https://server.undefinedstudio.fr:4443" }
$env:FABI_RELAY_TOKEN_FILE = $relayTokenFile
$env:FABI_NETWORK_IDENTITY_PATH = if ($env:FABI_NETWORK_IDENTITY_PATH) { $env:FABI_NETWORK_IDENTITY_PATH } else { Join-Path $networkState "worker.key" }
$env:FABI_FORCE_RELAY = if ($env:FABI_FORCE_RELAY) { $env:FABI_FORCE_RELAY } else { "0" }
$env:FABI_WORKER_SESSION_ID = [guid]::NewGuid().ToString()
$env:PARALLAX_KEY_PATH = Join-Path $HOME ".config\fabi\identity"
$env:PARALLAX_PROCESS_LOG_DIR = $processLogs
$env:PARALLAX_CUDA_SYSTEM_RESERVE_GB = "1.5"
$env:PYTHONPATH = (Join-Path $source "src")
$env:PYTHONUNBUFFERED = "1"
$env:RUST_LOG = if ($env:RUST_LOG) { $env:RUST_LOG } else { "info" }
$env:VLLM_ENGINE_READY_TIMEOUT_S = if ($env:VLLM_ENGINE_READY_TIMEOUT_S) { $env:VLLM_ENGINE_READY_TIMEOUT_S } else { "3600" }

Set-Location $source
& $python -m parallax.cli join `
  -s $schedulerEndpoint `
  -r `
  --max-batch-size 1 `
  --max-sequence-length 65536 `
  --max-num-tokens-per-batch 65536 `
  --kv-block-size 16 `
  --gpu-backend vllm `
  --tcp-port 19080 `
  --udp-port 19080 `
  --log-level DEBUG `
  1>> $outLog 2>> $errLog
exit $LASTEXITCODE
