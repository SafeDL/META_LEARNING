param(
    [string]$ResultRoot = "results/cutin_stage1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $root

$config = "mvr/configs/cutin_inner.yaml"
$prior = Join-Path $ResultRoot "interaction_prior.pt"
$priorGate = Join-Path $ResultRoot "interaction_prior_gate.json"
$context = Join-Path $ResultRoot "context_meta.pt"
$validation = Join-Path $ResultRoot "validation/report.json"

conda run -n metadrive python -m mvr.scripts.probe_frenet_contract `
    --output (Join-Path $ResultRoot "preflight.json")
if ($LASTEXITCODE -ne 0) { throw "Shared Frenet physical preflight failed" }

conda run -n metadrive python -m mvr.scripts.probe_cutin_risk_reachability `
    --config $config `
    --output (Join-Path $ResultRoot "risk_reachability.json")
if ($LASTEXITCODE -ne 0) { throw "Cut-in risk reachability preflight failed" }

conda run -n metadrive python -m mvr.scripts.train_mvr `
    --config $config `
    --output $ResultRoot `
    --stop-after interaction_prior
if ($LASTEXITCODE -ne 0) { throw "Interaction-prior training failed" }

conda run -n metadrive python -m mvr.scripts.evaluate_cutin_inner_training_gate `
    --config $config `
    --checkpoint $prior `
    --output $priorGate
if ($LASTEXITCODE -ne 0) { throw "Interaction-prior gate execution failed" }
if (-not ((Get-Content -LiteralPath $priorGate -Raw | ConvertFrom-Json).passed)) {
    throw "Interaction-prior gate did not pass; context_meta was not started"
}

conda run -n metadrive python -m mvr.scripts.train_mvr `
    --config $config `
    --output $ResultRoot `
    --resume $prior `
    --stop-after context_meta
if ($LASTEXITCODE -ne 0) { throw "Context-meta training failed" }

conda run -n metadrive python -m mvr.scripts.evaluate_cutin_inner_validation `
    --config $config `
    --checkpoint $context `
    --output $validation
if ($LASTEXITCODE -ne 0) { throw "Cut-in validation failed" }

conda run -n metadrive python -m mvr.scripts.render_cutin_inner_policy_gif `
    --config $config `
    --checkpoint $context `
    --validation $validation `
    --output-dir (Join-Path $ResultRoot "gif")
if ($LASTEXITCODE -ne 0) { throw "GIF rendering failed" }

conda run -n metadrive python -m mvr.scripts.plot_inner_sac_training `
    --manifest (Join-Path $ResultRoot "manifest.json") `
    --output (Join-Path $ResultRoot "training_curve.png")
if ($LASTEXITCODE -ne 0) { throw "Training-curve plotting failed" }

conda run -n metadrive python -m pytest mvr/tests -q
if ($LASTEXITCODE -ne 0) { throw "MVR tests failed" }
