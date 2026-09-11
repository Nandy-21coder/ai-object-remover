Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "c:\Users\user\Desktop\ai\ai-object-remover"
WshShell.Run "cmd.exe /c launch_app.bat /autostart", 0, False
