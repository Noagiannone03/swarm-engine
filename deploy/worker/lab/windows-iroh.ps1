$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$state = Join-Path $env:LOCALAPPDATA "fabi"
$runtime = Join-Path $state "runtime"
$manifestPath = Join-Path $state "MANIFEST"
$python = Join-Path $runtime "parallax-venv\Scripts\python.exe"
$sourcePointer = Join-Path $state "runtime-candidate-current.txt"
$schedulerPointer = Join-Path $state "scheduler-endpoint.txt"
$source = if ($env:FABI_PARALLAX_SOURCE) {
  $env:FABI_PARALLAX_SOURCE
} else {
  Join-Path $runtime "parallax-src"
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
$registryRoot = if ($env:FABI_MODEL_REGISTRY_ROOT) {
  $env:FABI_MODEL_REGISTRY_ROOT
} else {
  Join-Path $state "trust\model-registry-root-c0fe1ff1c8a45b286056f05d83e38039b8dd3e743e4d3e739177108b7d44285b.json"
}
$catalogBootstraps = if ($env:FABI_CATALOG_DHT_BOOTSTRAPS) {
  $env:FABI_CATALOG_DHT_BOOTSTRAPS
} else {
  @(
    "/ip4/37.59.98.16/tcp/19192/p2p/12D3KooWMQrc1rWXwaeQcshtANiw9FyyGWmfqAnVsStGRqsJ54Yi"
    "/ip4/37.59.98.16/tcp/19193/p2p/12D3KooWG8jJaC1upci3eDZ7XSobTPC5hT6bdqFGzzptH8q7b1eG"
  ) | ConvertTo-Json -Compress
}

function Read-RuntimeManifest {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "runtime manifest is not readable: $Path"
  }
  $values = @{}
  foreach ($line in Get-Content -LiteralPath $Path) {
    if ($line -match '^([^=]+)=(.*)$') {
      $values[$matches[1].Trim()] = $matches[2].Trim()
    }
  }
  return $values
}

$manifest = Read-RuntimeManifest -Path $manifestPath
if ($manifest.execution_engine -ne "skippy") {
  throw "unsupported installed execution engine: $($manifest.execution_engine)"
}
$executionDevice = $manifest.execution_device
if ($executionDevice -notin @("cuda", "vulkan")) {
  throw "unsupported Windows Skippy execution device: $executionDevice"
}

if (-not (Test-Path $python -PathType Leaf)) { throw "worker Python not found: $python" }
if (-not (Test-Path (Join-Path $source "pyproject.toml") -PathType Leaf)) { throw "invalid Parallax source: $source" }
if ($schedulerEndpoint -notmatch "^[0-9a-f]{64}$") { throw "invalid Iroh scheduler endpoint" }
if (-not (Test-Path $accountTokenFile -PathType Leaf)) { throw "account token file is not readable" }
if (-not (Test-Path $registryRoot -PathType Leaf)) { throw "pinned model-registry root is not readable" }

$networkState = Join-Path $state "network"
$processLogs = Join-Path $state "process-logs"
$outLog = Join-Path $state "worker-windows-iroh.out.log"
$errLog = Join-Path $state "worker-windows-iroh.err.log"
New-Item -ItemType Directory -Force -Path $networkState, $processLogs | Out-Null

$env:FABI_ACCOUNT_TOKEN = (Get-Content $accountTokenFile -Raw).Trim()
$env:FABI_NETWORK_TRANSPORT = "iroh"
$env:FABI_RELAY_URL = if ($env:FABI_RELAY_URL) { $env:FABI_RELAY_URL } else { "https://server.undefinedstudio.fr:4443" }
$env:FABI_RELAY_ENROLLMENT_URL = if ($env:FABI_RELAY_ENROLLMENT_URL) { $env:FABI_RELAY_ENROLLMENT_URL } else { "https://server.undefinedstudio.fr/fabi-registry/v1/network/enroll" }
Remove-Item Env:\FABI_RELAY_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:\FABI_RELAY_TOKEN_FILE -ErrorAction SilentlyContinue
$env:FABI_NETWORK_IDENTITY_PATH = if ($env:FABI_NETWORK_IDENTITY_PATH) { $env:FABI_NETWORK_IDENTITY_PATH } else { Join-Path $networkState "worker.key" }
$modelState = Join-Path $state "swarm-v3\qwen3-0-6b-v3"
$env:FABI_SWARM_V3_STATE_DIR = if ($env:FABI_SWARM_V3_STATE_DIR) { $env:FABI_SWARM_V3_STATE_DIR } else { Join-Path $modelState "registry" }
$env:FABI_SWARM_V3_FENCE_DB = if ($env:FABI_SWARM_V3_FENCE_DB) { $env:FABI_SWARM_V3_FENCE_DB } else { Join-Path $modelState "fencing.sqlite3" }
$env:FABI_SWARM_V3_MODE = if ($env:FABI_SWARM_V3_MODE) { $env:FABI_SWARM_V3_MODE } else { "active" }
$env:FABI_SWARM_V3_PLACEMENT = if ($env:FABI_SWARM_V3_PLACEMENT) { $env:FABI_SWARM_V3_PLACEMENT } else { "autonomous" }
$env:FABI_SWARM_V3_COORDINATION_MODE = "client"
$env:FABI_MODEL_REGISTRY_ROOT = $registryRoot
$env:FABI_MODEL_REGISTRY_METADATA_URL = if ($env:FABI_MODEL_REGISTRY_METADATA_URL) { $env:FABI_MODEL_REGISTRY_METADATA_URL } else { "https://server.undefinedstudio.fr/fabi-swarm-registry-v3/metadata/" }
$env:FABI_MODEL_REGISTRY_TARGETS_URL = if ($env:FABI_MODEL_REGISTRY_TARGETS_URL) { $env:FABI_MODEL_REGISTRY_TARGETS_URL } else { "https://server.undefinedstudio.fr/fabi-swarm-registry-v3/targets/" }
$env:FABI_CATALOG_DHT_MODE = if ($env:FABI_CATALOG_DHT_MODE) { $env:FABI_CATALOG_DHT_MODE } else { "client" }
$env:FABI_CATALOG_DHT_BOOTSTRAPS = $catalogBootstraps
$env:FABI_CATALOG_DHT_IDENTITY_PATH = if ($env:FABI_CATALOG_DHT_IDENTITY_PATH) { $env:FABI_CATALOG_DHT_IDENTITY_PATH } else { Join-Path $networkState "worker-catalog.key" }
$env:FABI_CATALOG_DHT_LISTEN_ADDRESS = if ($env:FABI_CATALOG_DHT_LISTEN_ADDRESS) { $env:FABI_CATALOG_DHT_LISTEN_ADDRESS } else { "/ip4/127.0.0.1/tcp/0" }
$env:FABI_FORCE_RELAY = if ($env:FABI_FORCE_RELAY) { $env:FABI_FORCE_RELAY } else { "0" }
$env:FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS = if ($env:FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS) { $env:FABI_INITIAL_ALLOCATION_TIMEOUT_SECONDS } else { "0" }
$env:FABI_WORKER_SESSION_ID = [guid]::NewGuid().ToString()
$env:PARALLAX_KEY_PATH = Join-Path $state "identity"
$env:PARALLAX_PROCESS_LOG_DIR = $processLogs
Remove-Item Env:\PARALLAX_CUDA_SYSTEM_RESERVE_GB -ErrorAction SilentlyContinue
$env:PYTHONPATH = (Join-Path $source "src")
$env:PYTHONUNBUFFERED = "1"
$env:RUST_LOG = if ($env:RUST_LOG) { $env:RUST_LOG } else { "info" }
$env:VLLM_ENGINE_READY_TIMEOUT_S = if ($env:VLLM_ENGINE_READY_TIMEOUT_S) { $env:VLLM_ENGINE_READY_TIMEOUT_S } else { "3600" }
New-Item -ItemType Directory -Force -Path $env:FABI_SWARM_V3_STATE_DIR, (Split-Path -Parent $env:FABI_SWARM_V3_FENCE_DB) | Out-Null

Set-Location $source
& $python -m parallax.cli join `
  -s $schedulerEndpoint `
  -r `
  --max-batch-size 1 `
  --max-sequence-length 32768 `
  --max-num-tokens-per-batch 8192 `
  --kv-block-size 32 `
  --gpu-backend skippy `
  --execution-device $executionDevice `
  --tcp-port 19080 `
  --udp-port 19080 `
  --log-level DEBUG `
  1>> $outLog 2>> $errLog
exit $LASTEXITCODE
