<#
.SYNOPSIS
    Instructor Test Case Runner for Tomato Leaf Disease Classification API.
.DESCRIPTION
    Sends images to the running FastAPI serving endpoint, displays prediction,
    confidence, latency, HTTP status code, and MLflow model version.
.PARAMETER ImagePath
    Path to a single image file to test.
.PARAMETER ImageDir
    Path to a directory containing image files to test.
.PARAMETER Port
    Port number where FastAPI serving is running (default: 8010).
.EXAMPLE
    .\scripts\run_instructor_testcase.ps1 -ImagePath dataset/processed_sample/test/Tomato___healthy/image-000.png
    .\scripts\run_instructor_testcase.ps1 -ImageDir dataset/processed_sample/test/Tomato___healthy -Port 8010
#>
[CmdletBinding()]
param (
    [string]$ImagePath = "",
    [string]$ImageDir = "",
    [int]$Port = 8010
)

$ErrorActionPreference = "Continue"
$BaseUrl = "http://127.0.0.1:$Port"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Tomato Leaf Disease Classifier - Instructor Test Suite  " -ForegroundColor Cyan
Write-Host "   Endpoint: $BaseUrl" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Service Health
try {
    $healthResp = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec 5
    $modelName = $healthResp.model_name
    $modelAlias = if ($healthResp.model_alias) { $healthResp.model_alias } else { "champion" }
    $modelVersion = if ($healthResp.model_version) { "v$($healthResp.model_version)" } else { "@$modelAlias" }
    Write-Host "[OK] Service is Healthy. Loaded Model: $modelName ($modelVersion)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Cannot connect to $BaseUrl/health. Is the serving process running?" -ForegroundColor Red
    Write-Host "Please start the serving service on port $Port before running this script." -ForegroundColor Yellow
    exit 1
}

# 2. Collect Images
$filesToTest = @()
if ($ImagePath -ne "") {
    if (Test-Path $ImagePath) {
        $filesToTest += (Get-Item $ImagePath)
    } else {
        Write-Host "[ERROR] ImagePath not found: $ImagePath" -ForegroundColor Red
        exit 1
    }
} elseif ($ImageDir -ne "") {
    if (Test-Path $ImageDir) {
        $filesToTest += Get-ChildItem -Path $ImageDir -File | Where-Object { $_.Extension -match '\.(jpg|jpeg|png)$' }
    } else {
        Write-Host "[ERROR] ImageDir not found: $ImageDir" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] No -ImagePath or -ImageDir provided. Auto-selecting sample images from dataset/tomato/test..." -ForegroundColor Yellow
    $sampleClasses = @("Tomato__Tomato_Healthy", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus")
    foreach ($cls in $sampleClasses) {
        $matches = Get-ChildItem -Path "dataset/tomato/test/$cls/*.jpg" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($matches) {
            $filesToTest += $matches
        }
    }
    if ($filesToTest.Count -eq 0) {
        $filesToTest += Get-ChildItem -Path "dataset" -Recurse -Filter "*.jpg" -ErrorAction SilentlyContinue | Select-Object -First 3
    }
}

if ($filesToTest.Count -eq 0) {
    Write-Host "[ERROR] No test images found to run." -ForegroundColor Red
    exit 1
}

Write-Host "`nTesting $($filesToTest.Count) images against /predict endpoint...`n" -ForegroundColor Cyan

# 3. Execute Predictions and Build Table
$results = @()

foreach ($file in $filesToTest) {
    $filePath = $file.FullName
    $fileName = $file.Name
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    
    $statusCode = 0
    $predictedClass = "N/A"
    $confidence = 0.0
    $errorDetail = ""
    
    try {
        # Use curl.exe for reliable multipart file upload on Windows PowerShell
        $curlOut = & curl.exe -s -w "`n%{http_code}" -X POST "$BaseUrl/predict" -F "file=@$filePath"
        $sw.Stop()
        $latencyMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 2)
        
        $lines = $curlOut -split "`n"
        if ($lines.Count -ge 2) {
            $httpCodeStr = $lines[-1].Trim()
            $bodyStr = ($lines[0..($lines.Count - 2)]) -join "`n"
            $statusCode = [int]$httpCodeStr
            
            if ($statusCode -eq 200) {
                $json = $bodyStr | ConvertFrom-Json
                $predictedClass = if ($json.disease_class) { $json.disease_class } elseif ($json.predicted_class) { $json.predicted_class } else { "Unknown" }
                $confRaw = if ($null -ne $json.confidence_score) { $json.confidence_score } elseif ($null -ne $json.confidence) { $json.confidence } else { 0.0 }
                $confidence = [math]::Round([double]$confRaw, 4)
            } else {
                $predictedClass = "[HTTP $statusCode ERROR]"
                $errorDetail = $bodyStr
            }
        } else {
            $statusCode = 500
            $predictedClass = "[PARSE ERROR]"
        }
    } catch {
        $sw.Stop()
        $latencyMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 2)
        $statusCode = 500
        $predictedClass = "[EXCEPTION]"
        $errorDetail = $_.Exception.Message
    }
    
    $results += [PSCustomObject]@{
        "File"            = $fileName
        "Status"          = $statusCode
        "Prediction"      = $predictedClass
        "Confidence"      = if ($confidence -gt 0) { "{0:P2}" -f $confidence } else { "N/A" }
        "Latency(ms)"     = $latencyMs
        "Model Version"   = "$modelVersion"
    }
}

$results | Format-Table -AutoSize

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Summary: $($results.Count) images tested. All responses logged." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan
