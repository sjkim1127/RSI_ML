param(
    [switch]$UseAutotune = $true,
    [int]$Tile = 0
)

$ErrorActionPreference = "Stop"

Write-Host "Building release binaries..."
cargo build --release -p rsi_ml

if ($Tile -gt 0) {
    $env:RSI_ML_MATMUL_TILE = "$Tile"
    Remove-Item Env:RSI_ML_MATMUL_AUTOTUNE -ErrorAction SilentlyContinue
    Write-Host "Using fixed tile: $Tile"
} elseif ($UseAutotune) {
    $env:RSI_ML_MATMUL_AUTOTUNE = "1"
    Remove-Item Env:RSI_ML_MATMUL_TILE -ErrorAction SilentlyContinue
    Write-Host "Using autotune tile selection."
}

Write-Host "Running matmul benchmark example..."
cargo run --release -p rsi_ml --example matmul_bench

Write-Host "Running criterion benchmark..."
cargo bench -p rsi_ml --bench matmul -- --noplot

Write-Host "Done. See target/criterion for reports."
