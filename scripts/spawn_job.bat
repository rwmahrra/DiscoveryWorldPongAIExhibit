
:: Black magic to save git branch and remote to %BRANCH% and %REMOTE%
FOR /F "tokens=*" %%g IN ('git rev-parse --abbrev-ref HEAD') do (SET BRANCH=%%g)
FOR /F "tokens=*" %%g IN ('git config --get remote.origin.url') do (SET REMOTE=%%g)

SET DATETIME = %date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%

SET ID=%BRANCH%_%DATETIME%
echo %BRANCH%
copy scripts\setup_template.sh scripts\temp.sh

:: Replace templating
powershell -Command "(gc scripts\temp.sh) -replace '{{BRANCH}}', '%BRANCH%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{REMOTE}}', '%REMOTE%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{ID}}', '%ID%' | Out-File -encoding ASCII scripts\temp.sh"

putty.exe -ssh neuwirtha@dh-ood.hpc.msoe.edu -pw %ROSIE_ACCESS% -m scripts\temp.sh
del scripts\temp.sh

