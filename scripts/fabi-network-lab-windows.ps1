param(
  [switch]$ForceRelay
)

$ErrorActionPreference = "Stop"
$Root = Join-Path $env:LOCALAPPDATA "fabi\network-lab"
$Binary = Join-Path $Root "fabi-network.exe"
$Identity = Join-Path $Root "windows.key"
$TokenFile = Join-Path $Root "relay.env"
$Log = Join-Path $Root "server.log"

$TokenLine = Get-Content -Path $TokenFile | Where-Object {
  $_.StartsWith("IROH_RELAY_ACCESS_TOKEN=")
} | Select-Object -First 1
if (-not $TokenLine) {
  throw "Missing IROH_RELAY_ACCESS_TOKEN in $TokenFile"
}
$Token = $TokenLine.Substring("IROH_RELAY_ACCESS_TOKEN=".Length).Trim()
if (-not $Token) {
  throw "Empty relay token in $TokenFile"
}

$env:FABI_RELAY_TOKEN = $Token
$Arguments = @(
  "--identity", $Identity,
  "--relay-url", "https://server.undefinedstudio.fr:4443"
)
if ($ForceRelay) {
  $Arguments += "--force-relay"
}
$Arguments += "serve"

& $Binary @Arguments *> $Log
