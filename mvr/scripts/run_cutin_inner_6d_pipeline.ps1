param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,
    [string]$ResultRoot = "results/cutin_stage1_6d_fewshot"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root

$resultRoot = Join-Path $root $ResultRoot
$priorCheckpoint = Join-Path $resultRoot "interaction_prior.pt"
$priorGate = Join-Path $resultRoot "interaction_prior_gate.json"
$checkpoint = Join-Path $resultRoot "context_meta.pt"
$config = "mvr/configs/cutin_inner.yaml"

while (-not (Test-Path -LiteralPath $priorCheckpoint)) {
    if (-not (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue)) {
        throw "Interaction-prior training ended without producing $priorCheckpoint"
    }
    Start-Sleep -Seconds 30
}

conda run -n metadrive python -m mvr.scripts.evaluate_cutin_inner_training_gate `
    --config $config `
    --checkpoint $priorCheckpoint `
    --output (Join-Path $ResultRoot "interaction_prior_gate.json")
if ($LASTEXITCODE -ne 0) { throw "Interaction-prior gate failed with exit code $LASTEXITCODE" }
if (-not ((Get-Content -LiteralPath $priorGate -Raw | ConvertFrom-Json).passed)) {
    throw "Interaction-prior gate did not pass"
}

conda run -n metadrive python -m mvr.scripts.train_mvr `
    --config $config `
    --output $ResultRoot `
    --resume $priorCheckpoint `
    --stop-after context_meta
if ($LASTEXITCODE -ne 0) { throw "Context-meta training failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Context-meta training ended without producing $checkpoint"
}

conda run -n metadrive python -m mvr.scripts.evaluate_cutin_inner_validation `
    --config $config `
    --checkpoint $checkpoint `
    --output (Join-Path $ResultRoot "validation/fixed_x0_k_comparison.json")
if ($LASTEXITCODE -ne 0) { throw "Validation failed with exit code $LASTEXITCODE" }

conda run -n metadrive python -m mvr.scripts.render_cutin_inner_policy_gif `
    --config $config `
    --checkpoint $checkpoint `
    --output-dir (Join-Path $ResultRoot "gif")
if ($LASTEXITCODE -ne 0) { throw "GIF rendering failed with exit code $LASTEXITCODE" }

conda run -n metadrive python -m mvr.scripts.plot_inner_sac_training `
    --manifest (Join-Path $ResultRoot "manifest.json") `
    --output (Join-Path $ResultRoot "inner_sac_training_curve.png")
if ($LASTEXITCODE -ne 0) { throw "Training-curve plotting failed with exit code $LASTEXITCODE" }

conda run -n metadrive python -m pytest `
    mvr/tests/test_training_contracts.py `
    mvr/tests/test_cutin_traffic_contract.py `
    mvr/tests/test_stage1_sampling.py `
    mvr/tests/test_cutin_inner_experiment.py `
    mvr/tests/test_failure_and_evaluation.py -q
if ($LASTEXITCODE -ne 0) { throw "Focused MVR tests failed with exit code $LASTEXITCODE" }
