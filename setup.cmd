@echo off
REM NexaRAG setup for Windows. Double-click this, or run it from cmd.exe.
REM
REM -ExecutionPolicy Bypass applies to this one process only and changes nothing about the
REM machine's policy. Without it, the default RemoteSigned policy refuses to run a .ps1
REM that came out of a git clone, with an error about the file not being digitally signed
REM — which reads as a security problem rather than a default.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" %*
endlocal
