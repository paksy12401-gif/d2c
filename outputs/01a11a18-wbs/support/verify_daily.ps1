$ErrorActionPreference='Stop'
$excel=$null
$book=$null
try {
  $excel=New-Object -ComObject Excel.Application
  $excel.Visible=$false
  $excel.DisplayAlerts=$false
  $excel.AutomationSecurity=3
  $book=$excel.Workbooks.Open((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\MDA_WBS_20261006-20261106.xlsx')).Path,0,$true)
  $excel.CalculateFull()
  $sheet=$book.Worksheets.Item('WBS')
  $first=([datetime]'2026-10-06').ToOADate()
  $last=([datetime]'2026-11-06').ToOADate()
  for($i=0;$i -lt 32;$i++) {
    if($sheet.Cells.Item(7,9+$i).Value2 -ne ($first+$i)) {throw "Date header mismatch at day $i"}
  }
  if($sheet.Range('AN7').Value2 -ne $last) {throw 'Last date mismatch'}
  if($sheet.Range('H3').Value2 -ne 1) {throw 'Not a one-day interval'}
  foreach($r in 9..47) {
    $start=$sheet.Cells.Item($r,5).Value2
    $end=$sheet.Cells.Item($r,6).Value2
    if($null -ne $start -and ($start -lt $first -or $end -gt $last -or $start -gt $end)) {throw "Task dates out of bounds at row $r"}
  }
  $blue=84+141*256+212*65536
  if($sheet.Range('I9').DisplayFormat.Interior.Color -ne $blue) {throw 'Initial task color missing'}
  if($sheet.Range('AN47').DisplayFormat.Interior.Color -ne $blue) {throw 'Final date not colored'}
  $sheet.Range('E9').Value2=[double]($first+1)
  $sheet.Range('F9').Value2=[double]($first+1)
  $excel.CalculateFull()
  if($sheet.Range('I9').DisplayFormat.Interior.Color -eq $blue) {throw 'Old color did not clear'}
  if($sheet.Range('J9').DisplayFormat.Interior.Color -ne $blue) {throw 'One-day date change did not move the bar one column'}
  if($sheet.Range('K9').DisplayFormat.Interior.Color -eq $blue) {throw 'Task color spilled past one day'}
  $sheet.Range('E9').Value2=[double]($first+4)
  $sheet.Range('F9').Value2=[double]($first+5)
  $excel.CalculateFull()
  if($sheet.Range('M9').DisplayFormat.Interior.Color -ne $blue -or $sheet.Range('N9').DisplayFormat.Interior.Color -ne $blue) {throw 'Weekend dates missing color'}
  if($sheet.Range('G9').Value2 -ne 2) {throw 'Inclusive duration incorrect'}
  Write-Output 'PASS: 32 daily headers, all task dates in requested range, final date colored, one-day edits move exactly one column, weekends included.'
} finally {
  if($book) {$book.Close($false)}
  if($excel) {$excel.Quit();[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excel)}
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}
