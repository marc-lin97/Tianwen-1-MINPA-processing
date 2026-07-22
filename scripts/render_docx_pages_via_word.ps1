param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class WordWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@

$inputFull = (Resolve-Path -LiteralPath $InputPath).Path
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputFull = (Resolve-Path -LiteralPath $OutputDirectory).Path

$wdGoToPage = 1
$wdGoToAbsolute = 1
$wdStatisticPages = 2
$word = New-Object -ComObject Word.Application
$word.Visible = $true
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3

try {
    $document = $word.Documents.Open($inputFull, $false, $true)
    $document.Activate()
    $word.ActiveWindow.WindowState = 1
    $word.ActiveWindow.View.Type = 3
    $word.ActiveWindow.View.Zoom.PageFit = 1
    $hwnd = [IntPtr]$word.ActiveWindow.Hwnd
    [WordWindowCapture]::SetForegroundWindow($hwnd) | Out-Null
    $pageCount = $document.ComputeStatistics($wdStatisticPages)
    for ($page = 1; $page -le $pageCount; $page++) {
        $start = $document.GoTo($wdGoToPage, $wdGoToAbsolute, $page).Start
        if ($page -lt $pageCount) {
            $end = $document.GoTo($wdGoToPage, $wdGoToAbsolute, $page + 1).Start - 1
        } else {
            $end = $document.Content.End
        }
        $range = $document.Range($start, $end)
        $range.Select()
        $word.ActiveWindow.ScrollIntoView($range, $true)
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 350
        $rect = New-Object WordWindowCapture+RECT
        if (-not [WordWindowCapture]::GetWindowRect($hwnd, [ref]$rect)) {
            throw "Could not read the Word window bounds for page $page."
        }
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        $image = New-Object System.Drawing.Bitmap($width, $height)
        $outputPath = Join-Path $outputFull ('page-{0:D2}.png' -f $page)
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($image)
            $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $image.Size)
            $image.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
        } finally {
            if ($null -ne $graphics) { $graphics.Dispose() }
            $image.Dispose()
        }
        Write-Output $outputPath
    }
    $document.Close($false)
} finally {
    $word.Quit()
}
