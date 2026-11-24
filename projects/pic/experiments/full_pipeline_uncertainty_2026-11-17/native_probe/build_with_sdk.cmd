@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /O2 /LD /fp:strict /std:c17 selected_inverse.c /Fe:selected_inverse.dll /Fo:selected_inverse.obj /link /OUT:selected_inverse.dll
