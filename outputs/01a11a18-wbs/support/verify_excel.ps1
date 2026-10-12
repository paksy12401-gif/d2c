$ErrorActionPreference = 'Stop'
$excel = $null
$book = $null
$checks = @()
function Assert-Check($label, $condition) {
    if (-not $condition) { throw "Check failed: $label" }
    $script:checks += $label
}
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.AutomationSecurity = 3
    $taskFile = Join-Path $PSScriptRoot '..\MDA_WBS_수정본.xlsx'
    $book = $excel.Workbooks.Open((Resolve-Path -LiteralPath $taskFile).Path, 0, $true)
    $excel.Calculation = -4105
    $excel.CalculateFull()
    $sheet = $book.Worksheets.Item('WBS')
    $blue = 84 + 141 * 256 + 212 * 65536
    $pink = 217 + 149 * 256 + 148 * 65536
    $baseStart = $sheet.Range('E9').Value2
    $baseEnd = $sheet.Range('F9').Value2
    Assert-Check 'Initial task bar colored' ($sheet.Range('I9').DisplayFormat.Interior.Color -eq $blue)
    Assert-Check 'Initial later cell uncolored' ($sheet.Range('K9').DisplayFormat.Interior.Color -ne $blue)
    Assert-Check 'Inclusive duration = 1' ($sheet.Range('G9').Value2 -eq 1)
    Assert-Check 'Date typo corrected' ($sheet.Range('E29').Value2 -eq $baseStart)
    $sheet.Range('E9').Value2 = [double]($baseStart + 4)
    $sheet.Range('F9').Value2 = [double]($baseStart + 4)
    $excel.CalculateFull()
    Assert-Check 'Old bar cleared after date change' ($sheet.Range('I9').DisplayFormat.Interior.Color -ne $blue)
    Assert-Check 'Weekend task moved to new interval' ($sheet.Range('K9').DisplayFormat.Interior.Color -eq $blue)
    $sheet.Range('C9').Value2 = '김성준'
    $excel.CalculateFull()
    Assert-Check 'Owner color reacts to edit' ($sheet.Range('K9').DisplayFormat.Interior.Color -eq $pink)
    $sheet.Range('C9').Value2 = '김성준, 문홍식'
    $excel.CalculateFull()
    Assert-Check 'New joint assignment receives shared color' ($sheet.Range('K9').DisplayFormat.Interior.Color -eq (182 + 215 * 256 + 168 * 65536))
    $sheet.Range('C9').Value2 = '김성준'
    $sheet.Range('E9').ClearContents()
    $excel.CalculateFull()
    Assert-Check 'Blank date clears task bar' ($sheet.Range('K9').DisplayFormat.Interior.Color -ne $pink)
    Assert-Check 'Blank date clears duration' ($sheet.Range('G9').Value2 -eq '')
    $sheet.Range('E9').Value2 = [double]($baseStart + 5)
    $sheet.Range('F9').Value2 = [double]($baseStart + 4)
    $excel.CalculateFull()
    Assert-Check 'Reversed dates flagged' ($sheet.Range('G9').Text -eq '날짜 확인')
    Assert-Check 'Reversed dates do not fill bar' ($sheet.Range('K9').DisplayFormat.Interior.Color -ne $pink)
    $sheet.Range('E9').Value2 = [double]$baseStart
    $sheet.Range('F9').Value2 = [double]$baseEnd
    $sheet.Range('C9').Value2 = '팀원 전체'
    $baseLast = $sheet.Range('F47').Value2
    $sheet.Range('F47').Value2 = [double]($baseLast + 150)
    $excel.CalculateFull()
    $lastHeader = $sheet.Range('AQ7').Value2
    $step = $sheet.Range('H3').Value2
    Assert-Check 'Extended schedule adjusts interval' ($step -gt 2)
    $dateValues = @($sheet.Range('I7:AQ7').Value2)
    $covered = $false
    for ($c = 9; $c -le 43; $c++) {
        $d = $sheet.Cells.Item(7,$c).Value2
        if (($d -is [double]) -and $d -le ($baseLast + 150) -and ($d + $step - 1) -ge ($baseLast + 150)) {
            $covered = $sheet.Cells.Item(47,$c).DisplayFormat.Interior.Color -eq $blue
        }
    }
    Assert-Check 'Extended end date remains visible and colored' $covered
    $sheet.Range('F47').Value2 = [double]$baseLast
    $sheet.Range('E9').Value2 = [double]($baseStart - 60)
    $excel.CalculateFull()
    Assert-Check 'Earlier start shifts timeline' ($sheet.Range('I7').Value2 -eq ($baseStart - 60))
    $checks | ConvertTo-Json
} finally {
    if ($book) { $book.Close($false) }
    if ($excel) { $excel.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
