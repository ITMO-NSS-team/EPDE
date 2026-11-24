@echo off
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe" /nologo /O2 /c /GS- /fp:strict /std:c17 native_score.c /Fo:native_score.obj
if errorlevel 1 exit /b 1
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe" /NOLOGO /DLL /NOENTRY /NODEFAULTLIB /MACHINE:X64 native_score.obj /OUT:native_score.dll
