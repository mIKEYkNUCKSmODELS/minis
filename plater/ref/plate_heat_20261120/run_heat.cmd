@echo off
cd /d "C:\Users\confi\MKM Tools\Plate Packer"
set PYTHONIOENCODING=utf-8
"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe" -I -u "C:\MKM_TMP\plate_heat_20261120\plate_heat.py" %1 %2 %3 > "%~2\stdout.txt" 2>&1
echo done > "%~2\run.done"
