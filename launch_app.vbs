Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
thisFolder = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = thisFolder
WshShell.Run "cmd.exe /c launch_app.bat", 0, False
