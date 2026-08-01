param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Width = 1224,
    [int]$Height = 1584,
    [switch]$SkipPdfExport
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $InputDocx).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $output)) {
    [IO.Directory]::CreateDirectory($output) | Out-Null
}
if (Get-ChildItem -LiteralPath $output -File -ErrorAction SilentlyContinue) {
    throw "Output directory must be empty: $output"
}

$word = $null
$document = $null
$powerPoint = $null
$presentation = $null
$rendered = @()
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($source, $false, $true)
    $pdf = $null
    if (-not $SkipPdfExport) {
        $pdf = Join-Path $output 'document.pdf'
        $document.ExportAsFixedFormat($pdf, 17)
    }
    $pageCount = $document.ComputeStatistics(2)
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Add()
    $presentation.PageSetup.SlideWidth = $document.PageSetup.PageWidth
    $presentation.PageSetup.SlideHeight = $document.PageSetup.PageHeight

    for ($page = 1; $page -le $pageCount; $page++) {
        $start = $document.GoTo(1, 1, $page)
        if ($page -lt $pageCount) {
            $next = $document.GoTo(1, 1, $page + 1)
            $end = $next.Start - 1
        } else {
            $end = $document.Content.End
        }
        $range = $document.Range($start.Start, $end)
        # Avoid the Office clipboard: repeated CopyAsPicture/PasteSpecial calls
        # can deadlock in headless COM automation. Word exposes the same range
        # as EMF bytes, which PowerPoint can import deterministically.
        $emf = Join-Path $output ('page-{0:D3}.emf' -f $page)
        [IO.File]::WriteAllBytes($emf, $range.EnhMetaFileBits)
        $slide = $presentation.Slides.Add(1, 12)
        $shape = $slide.Shapes.AddPicture(
            $emf, 0, -1,
            $document.PageSetup.LeftMargin,
            $document.PageSetup.TopMargin,
            -1, -1
        )
        $shape.LockAspectRatio = -1
        $shape.Width = $document.PageSetup.PageWidth -
            $document.PageSetup.LeftMargin - $document.PageSetup.RightMargin
        $png = Join-Path $output ('page-{0:D3}.png' -f $page)
        $slide.Export($png, 'PNG', $Width, $Height)
        $file = Get-Item -LiteralPath $png
        $rendered += [ordered]@{
            page = $page
            path = $file.FullName
            bytes = $file.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $png).Hash.ToLower()
        }
        $slide.Delete()
        Remove-Item -LiteralPath $emf -Force
    }

    $manifest = [ordered]@{
        status = 'PASS'
        renderer = 'Microsoft Word read-only Range.EnhMetaFileBits + PowerPoint PNG export'
        source = $source
        source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLower()
        pdf = $pdf
        pdf_sha256 = if ($pdf) {
            (Get-FileHash -Algorithm SHA256 -LiteralPath $pdf).Hash.ToLower()
        } else { $null }
        page_count = $pageCount
        png_width = $Width
        png_height = $Height
        pages = $rendered
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
    }
    $manifestPath = Join-Path $output 'render_manifest.json'
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
    $manifest | ConvertTo-Json -Depth 5
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
    if ($null -ne $document) { $document.Close($false) }
    if ($null -ne $word) { $word.Quit() }
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}
