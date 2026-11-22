"""MKM change detector -- no AI, stdlib only.

Answers the driver check's step 1 ("anything new in the hub / AI_INBOX? PROGRESS tail?")
so the AI only has to read ONE small file, and can end at once when it says NO CHANGE.

Stateless: lists files modified after the driver's last ACK (or the last 20 min if no ACK).
The driver acks by running:  python change_detector.py --ack
Writes only to its own OUT_DIR. Reads everything else. Deletes nothing.
"""
import argparse
import datetime as dt
import os
import sys

HUB = r"G:\My Drive\MKM AI HUB"
INBOX = r"F:\MKM_STAGE\AI_INBOX"
PROGRESS = r"F:\MKM_STAGE\PROGRESS.txt"
OUT_DIR = r"F:\MKM_STAGE\WATCH"

# The driver's own files are not news to the driver.
OWN_PREFIXES = ("DRIVER", "TASK", "MKM BOARD")
OWN_DIRS = ("CARDS",)
DEFAULT_WINDOW_MIN = 20


def is_own(root, path):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts[0] in OWN_DIRS:
        return True
    return os.path.basename(path).upper().startswith(OWN_PREFIXES)


def scan(root, since, skip_own):
    found = []
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if m > since and not (skip_own and is_own(root, p)):
                found.append((m, os.path.relpath(p, root)))
    return sorted(found)


def tail(path, n=3):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return lines[-n:]
    except OSError as e:
        return [f"(cannot read: {e})"]


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=HUB)
    ap.add_argument("--inbox", default=INBOX)
    ap.add_argument("--progress", default=PROGRESS)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--ack", action="store_true", help="driver has read CHANGES.txt")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    ack_file = os.path.join(a.out, "ACK.txt")
    now = dt.datetime.now().timestamp()

    if a.ack:
        write_atomic(ack_file, f"{now}\n")
        print("ACK written")
        return 0

    try:
        with open(ack_file, encoding="utf-8") as f:
            since = float(f.read().strip())
    except (OSError, ValueError):
        since = now - DEFAULT_WINDOW_MIN * 60

    hub = scan(a.hub, since, skip_own=True)
    inbox = scan(a.inbox, since, skip_own=False)
    stamp = lambda t: dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    n = len(hub or []) + len(inbox or [])
    out = [f"{'CHANGED ' + str(n) if n else 'NO CHANGE'} since {stamp(since)} (checked {stamp(now)})"]
    for label, root, items in (("HUB", a.hub, hub), ("AI_INBOX", a.inbox, inbox)):
        if items is None:
            out.append(f"{label}: MISSING {root}")
            continue
        for m, rel in items:
            out.append(f"{label} {stamp(m)} {rel}")
    out.append("PROGRESS tail:")
    out += ["  " + l for l in tail(a.progress)]
    write_atomic(os.path.join(a.out, "CHANGES.txt"), "\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
