' Silent Background Starter for AI Object Remover Server
Option Explicit
Dim oShell, fso, thisFolder, backendBat, cmdLine
Set oShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

thisFolder = fso.GetParentFolderName(WScript.ScriptFullName)
backendBat = thisFolder & "\backend\run_backend.bat"
oShell.CurrentDirectory = fso.GetParentFolderName(backendBat)
cmdLine = "cmd.exe /c " & Chr(34) & Chr(34) & backendBat & Chr(34) & " > " & Chr(34) & thisFolder & "\backend\vbs_debug.log" & Chr(34) & " 2>&1" & Chr(34)
oShell.Run cmdLine, 0, False
