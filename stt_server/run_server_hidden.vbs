' Launch run_server.bat with no visible window.
'
' Why: schtasks /create /tr run_server.bat causes a cmd.exe window to
' appear whenever the task fires under an interactive user session. The
' window is easy to close by mistake, which kills the uvicorn child
' process and takes the GPU STT server down with it.
'
' Wscript.Shell.Run's second arg controls window state: 0 = hidden.
' Third arg "False" means don't wait for the .bat to finish — fire and
' return so the task scheduler doesn't think the task is still running.
'
' The .bat itself stays the same — it still cd's via %~dp0 and redirects
' uvicorn output to server.log. This .vbs is purely a window-hider.

Set objShell = WScript.CreateObject("WScript.Shell")
objShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
objShell.Run "run_server.bat", 0, False
