# DryRun-FixMojibake.ps1 — Detect mojibake in .py files (no modifications)

$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Files = Get-ChildItem -Path $Root -Recurse -Filter *.py

# Correct targets we want to protect/recover
$targets = @('→','–','—','‘','’','“','”','…','•')

# Encoders/decoders
$utf8  = [Text.Encoding]::UTF8
$win   = [Text.Encoding]::GetEncoding(1252)

# Build bad variants map: for each correct char, create 1x and 2x mojibake strings
$badStrings = New-Object System.Collections.Generic.List[string]
foreach ($t in $targets) {
    $bytes = $utf8.GetBytes($t)
    $bad1  = $win.GetString($bytes)            # one level mojibake (e.g., â†’)
    $bad2  = $win.GetString($utf8.GetBytes($bad1))  # double mojibake (e.g., Ă˘â€ â€™)
    $badStrings.Add($bad1)
    $badStrings.Add($bad2)
}

$hits = @()
foreach ($f in $Files) {
    $text = Get-Content $f.FullName -Raw
    foreach ($b in $badStrings) {
        if ([string]::IsNullOrEmpty($b)) { continue }
        if ($text.Contains($b)) {
            $hits += [pscustomobject]@{ File = $f.FullName; Pattern = $b }
        }
    }
}

if ($hits.Count -eq 0) {
    Write-Host "No mojibake patterns found." -ForegroundColor Green
} else {
    $hits | Group-Object File | ForEach-Object {
        Write-Host "`nFILE: $($_.Name)" -ForegroundColor Yellow
        $_.Group | Select-Object -ExpandProperty Pattern | Sort-Object -Unique | ForEach-Object { "  - $_" }
    }
    Write-Host "`nTotal files with issues: $(( $hits | Select-Object -Expand File | Sort-Object -Unique ).Count)"
}
