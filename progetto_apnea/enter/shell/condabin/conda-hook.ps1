$Env:CONDA_EXE = "/Users/domenicodisivo/Documents/apps/fl_apnea/progetto_apnea/enter/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/Users/domenicodisivo/Documents/apps/fl_apnea/progetto_apnea/enter"
$Env:_CONDA_EXE = "/Users/domenicodisivo/Documents/apps/fl_apnea/progetto_apnea/enter/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $False}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs