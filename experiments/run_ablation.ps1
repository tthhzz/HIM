$ErrorActionPreference = "Stop"

$datasetPath = "data/locomo10.json"
$modelName = "qwen2.5:3b"
$backendName = "ollama"
$sampleRatio = 0.1
$retrieveK = 10
$outputDir = "results/runs"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Invoke-HimExperiment {
    param(
        [string]$Name,
        [bool]$SourceMonitoring,
        [bool]$AttentionGating,
        [bool]$Consolidation
    )

    Write-Host "Running $Name" -ForegroundColor Cyan
    python experiments/evaluate_him.py `
        --dataset $datasetPath `
        --model $modelName `
        --backend $backendName `
        --ratio $sampleRatio `
        --retrieve_k $retrieveK `
        --enable_source_monitoring $SourceMonitoring.ToString().ToLowerInvariant() `
        --enable_activation_retrieval true `
        --enable_attention_gating $AttentionGating.ToString().ToLowerInvariant() `
        --enable_consolidation $Consolidation.ToString().ToLowerInvariant() `
        --output "$outputDir/$Name.json"
}

Invoke-HimExperiment -Name "him_full" -SourceMonitoring $true -AttentionGating $true -Consolidation $true
Invoke-HimExperiment -Name "him_without_encoding" -SourceMonitoring $false -AttentionGating $false -Consolidation $true
Invoke-HimExperiment -Name "him_without_consolidation" -SourceMonitoring $true -AttentionGating $true -Consolidation $false
