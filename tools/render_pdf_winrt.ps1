param(
    [Parameter(Mandatory=$true)][string]$InputPdf,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$Width = 1224,
    [int]$Height = 1584
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $InputPdf).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $output)) {
    [IO.Directory]::CreateDirectory($output) | Out-Null
}
if (Get-ChildItem -LiteralPath $output -File -ErrorAction SilentlyContinue) {
    throw "Output directory must be empty: $output"
}

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime]
$null = [Windows.Data.Pdf.PdfPageRenderOptions, Windows.Data.Pdf, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime]

function Await-Result($Operation, [Type]$ResultType) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } | Select-Object -First 1
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Await-Action($Operation) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq 'AsTask' -and -not $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } | Select-Object -First 1
    $task = $method.Invoke($null, @($Operation))
    $task.Wait()
}

$storageFile = Await-Result (
    [Windows.Storage.StorageFile]::GetFileFromPathAsync($source)
) ([Windows.Storage.StorageFile])
$pdf = Await-Result (
    [Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($storageFile)
) ([Windows.Data.Pdf.PdfDocument])

$rendered = @()
for ($index = 0; $index -lt $pdf.PageCount; $index++) {
    $page = $null
    $memory = $null
    $netStream = $null
    $fileStream = $null
    try {
        $page = $pdf.GetPage($index)
        $memory = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
        $options = New-Object Windows.Data.Pdf.PdfPageRenderOptions
        $options.DestinationWidth = $Width
        $options.DestinationHeight = $Height
        Await-Action ($page.RenderToStreamAsync($memory, $options))
        $netStream = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($memory)
        $png = Join-Path $output ('page-{0:D3}.png' -f ($index + 1))
        $fileStream = [IO.File]::Create($png)
        $netStream.CopyTo($fileStream)
        $fileStream.Close()
        $fileStream = $null
        $file = Get-Item -LiteralPath $png
        $rendered += [ordered]@{
            page = $index + 1
            path = $file.FullName
            bytes = $file.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $png).Hash.ToLower()
        }
    } finally {
        if ($fileStream) { $fileStream.Close() }
        if ($netStream) { $netStream.Close() }
        if ($memory) { $memory.Dispose() }
        if ($page) { $page.Dispose() }
    }
}

$manifest = [ordered]@{
    status = 'PASS'
    renderer = 'Windows.Data.Pdf PdfDocument.RenderToStreamAsync'
    source = $source
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLower()
    page_count = $pdf.PageCount
    png_width = $Width
    png_height = $Height
    pages = $rendered
    timestamp_utc = [DateTime]::UtcNow.ToString('o')
}
$manifestPath = Join-Path $output 'render_manifest.json'
[IO.File]::WriteAllText(
    $manifestPath,
    ($manifest | ConvertTo-Json -Depth 5),
    [Text.UTF8Encoding]::new($false)
)
$manifest | ConvertTo-Json -Depth 5
