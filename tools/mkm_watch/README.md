# MKM change detector (no AI)

Does the driver check's step 1 for free: every 5 min, Windows Task Scheduler runs
`change_detector.py`, which writes `F:\MKM_STAGE\WATCH\CHANGES.txt`:

- line 1: `NO CHANGE since ...` or `CHANGED n since ...`
- new/changed files in the hub (driver's own DRIVER*/TASK*/MKM BOARD*/CARDS skipped) and in AI_INBOX
- last 3 lines of PROGRESS.txt

Stdlib Python only. Reads everything, writes only to `F:\MKM_STAGE\WATCH\`, deletes nothing,
uses a few MB of RAM for a second.

## Install (on MKM)
1. Copy `change_detector.py` to `F:\MKM_STAGE\WATCH\change_detector.py`.
2. Check the `PROGRESS` path at the top of the script matches the real PROGRESS.txt.
3. Register (cmd, not PowerShell, so the execution policy stays Restricted):
   `schtasks /create /tn MKM_ChangeDetector /sc minute /mo 5 /tr "pythonw F:\MKM_STAGE\WATCH\change_detector.py" /f`
4. Undo: `schtasks /delete /tn MKM_ChangeDetector /f` (then the WATCH folder can go; it is only this tool's scratch).

## Driver check, new step 1
Read `F:\MKM_STAGE\WATCH\CHANGES.txt` only. If line 1 is `NO CHANGE` and nothing is pending
on your side: reschedule and end the turn, no other tool calls. Otherwise act on the listed files,
then run `python F:\MKM_STAGE\WATCH\change_detector.py --ack`.
