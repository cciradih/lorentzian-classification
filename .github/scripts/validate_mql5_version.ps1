param(
  [string]$RequestedTag = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$versionPath = Join-Path $repoRoot "ports\mql5\VERSION"
$version = (Get-Content $versionPath -Raw).Trim()

if ($version -notmatch '^(0|[1-9]\d*)\.\d+$') {
  throw "ports/mql5/VERSION must use MQL5's X.Y format; got '$version'."
}

$sources = @(
  "ports\mql5\indicators\LorentzianClassification\LorentzianClassification.mq5",
  "ports\mql5\experts\LorentzianClassification_EA.mq5"
)
$propertyPattern = '(?m)^\s*#property\s+version\s+"([^"]+)"\s*$'

foreach ($relativeSource in $sources) {
  $sourcePath = Join-Path $repoRoot $relativeSource
  $matches = [regex]::Matches((Get-Content $sourcePath -Raw), $propertyPattern)
  if ($matches.Count -ne 1) {
    throw "$relativeSource must contain exactly one #property version declaration."
  }

  $embeddedVersion = $matches[0].Groups[1].Value
  if ($embeddedVersion -cne $version) {
    throw "$relativeSource embeds version '$embeddedVersion', but ports/mql5/VERSION is '$version'."
  }
}

$tag = "mql5-v$version"
if ($RequestedTag -and $RequestedTag -cne $tag) {
  throw "Requested tag '$RequestedTag' does not match canonical tag '$tag'."
}

if ($env:GITHUB_OUTPUT) {
  Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "version=$version"
  Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "tag=$tag"
}

Write-Host "Validated MQL5 version $version ($tag) in VERSION, indicator, and EA."
