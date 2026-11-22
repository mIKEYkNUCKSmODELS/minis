"""Archive-aware piece paths (Mike 0036): a piece path is a normal file OR '<archive>::<member>'.

    F:\\X.zip::Folder/part.stl      .zip via zipfile (member streamed into memory, nothing written to disk)
    F:\\X.7z::Folder/part.stl       .7z/.rar/.tar... via 7-Zip CLI `7z e -so` (stdout stream); unavailable if 7z is missing
Archives are opened read-only and are never modified. '::' cannot occur in a Windows drive path ('F:\\'), so the split is unambiguous.
"""
import hashlib
import os
import re
import subprocess
import zipfile

SEP = "::"
_ARCH = (".zip", ".7z", ".rar", ".tar", ".gz", ".tgz")
_SEVENZ = [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]
_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.I)


class ArchiveError(ValueError):
    pass


def split_path(p):
    """-> (archive, member) or (path, None)."""
    p = str(p)
    i = p.find(SEP)
    if i > 0 and p[:i].lower().endswith(_ARCH):
        return p[:i], p[i + len(SEP):].replace("\\", "/")
    return p, None


def is_archive_path(p):
    return split_path(p)[1] is not None


def norm(p):
    """Full-string normal form used by accounting: archive part normcased+absolute, member part kept exactly."""
    a, m = split_path(p)
    a = os.path.normcase(os.path.abspath(a))
    return a if m is None else a + SEP + m


def _7z():
    for c in _SEVENZ:
        if os.path.isfile(c):
            return c
    return None


def sevenzip_available():
    return _7z() is not None


def _is_zip(a):
    return a.lower().endswith(".zip")


def read_bytes(p):
    """Whole file or member as bytes. Names the archive and member in every refusal."""
    a, m = split_path(p)
    if m is None:
        with open(a, "rb") as fh:
            return fh.read()
    if not os.path.isfile(a):
        raise ArchiveError("archive missing: %s (member %s)" % (a, m))
    if _is_zip(a):
        try:
            with zipfile.ZipFile(a, "r") as zf:
                try:
                    info = zf.getinfo(m)
                except KeyError:
                    raise ArchiveError("member missing in archive: %s :: %s" % (a, m))
                try:
                    return zf.read(info)
                except (zipfile.BadZipFile, RuntimeError, OSError, EOFError) as e:
                    raise ArchiveError("member corrupt or unreadable: %s :: %s (%s)" % (a, m, e))
        except zipfile.BadZipFile as e:
            raise ArchiveError("archive corrupt: %s (%s)" % (a, e))
    z = _7z()
    if z is None:
        raise ArchiveError("7-Zip not installed, cannot read %s :: %s" % (a, m))
    r = subprocess.run([z, "e", "-so", "-y", a, m], capture_output=True, creationflags=0x08000000)
    if r.returncode != 0 or not r.stdout:
        raise ArchiveError("7z could not read member: %s :: %s (rc=%s %s)" % (a, m, r.returncode, r.stderr[-200:].decode("utf8", "replace")))
    return r.stdout


def list_members(archive):
    """[(member, size, crc32)] without extracting. .zip via zipfile; others via `7z l -slt`."""
    if not os.path.isfile(archive):
        raise ArchiveError("archive missing: %s" % archive)
    if _is_zip(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            return [(i.filename, i.file_size, i.CRC) for i in zf.infolist() if not i.is_dir()]
    z = _7z()
    if z is None:
        raise ArchiveError("7-Zip not installed, cannot list %s" % archive)
    out = subprocess.run([z, "l", "-slt", "-y", archive], capture_output=True, creationflags=0x08000000).stdout.decode("utf8", "replace").replace("\r\n", "\n")
    res = []
    for blk in out.split("\n\n"):
        d = dict(l.split(" = ", 1) for l in blk.splitlines() if " = " in l)
        if d.get("Path") and "Size" in d and d.get("Folder") != "+" and not d.get("Attributes", "").startswith("D"):
            res.append((d["Path"].replace("\\", "/"), int(d["Size"]), int(d["CRC"], 16) if d.get("CRC") else 0))
    return res


def size(p):
    a, m = split_path(p)
    if m is None:
        return os.path.getsize(a)
    for name, sz, _ in list_members(a):
        if name == m:
            return sz
    raise ArchiveError("member missing in archive: %s :: %s" % (a, m))


def sha256(p):
    """sha256 of the file, or of the MEMBER'S bytes (so a recipe hash is the same as for the unzipped file)."""
    a, m = split_path(p)
    h = hashlib.sha256()
    if m is None:
        with open(a, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
    else:
        h.update(read_bytes(p))
    return h.hexdigest()


_AHASH = {}


def cache_key(p):
    """Per-model cross-section cache key: archive SHA256 + member CRC for an archive member, else the file sha256."""
    a, m = split_path(p)
    if m is None:
        return sha256(p)
    st_ = os.stat(a); mk = (os.path.abspath(a), st_.st_size, st_.st_mtime_ns)      # an archive is hashed once per run, not once per member
    if mk not in _AHASH:
        ha = hashlib.sha256()
        with open(a, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                ha.update(blk)
        _AHASH[mk] = ha.hexdigest()
    crc = [c for n, _, c in list_members(a) if n == m]
    if not crc:
        raise ArchiveError("member missing in archive: %s :: %s" % (a, m))
    return "%s:%08x" % (_AHASH[mk], crc[0])


# ---- planner-facing helpers (Mike 0036 merge, 11/22): the planner (mkm_gpupool10_fill.py) reads pieces through these, so a path may be a file or '<archive>::<member>' ----
class _Stat:
    """os.stat-like record: size and mtime stamp.  For an archive member: size = member size, mtime_ns = archive mtime_ns, crc = the member CRC32 (content stamp without reading the member)."""
    def __init__(self, size, mtime_ns, crc=None):
        self.st_size = size; self.st_mtime_ns = mtime_ns; self.st_mtime = mtime_ns / 1e9; self.crc = crc


def stat(p):
    a, m = split_path(p)
    if m is None:
        return os.stat(a)
    if not os.path.isfile(a):
        raise ArchiveError("archive missing: %s (member %s)" % (a, m))
    st = os.stat(a)
    for name, sz, crc in _members_cached(a, st.st_mtime_ns):
        if name == m:
            return _Stat(sz, st.st_mtime_ns, crc)
    raise ArchiveError("member missing in archive: %s :: %s" % (a, m))


_LM = {}


def _members_cached(a, mtime_ns):
    k = (a, mtime_ns)
    if k not in _LM:
        _LM[k] = list_members(a)
    return _LM[k]


def abspath(p):
    """os.path.abspath for a file; for an archive member: abspath of the archive + '::' + the member exactly as written ('/' separators)"""
    a, m = split_path(p)
    return os.path.abspath(a) if m is None else os.path.abspath(a) + SEP + m


def open_rb(p):
    """binary file object: the real file, or the member's bytes in memory (nothing is written to disk)"""
    import io
    a, m = split_path(p)
    return open(a, "rb") if m is None else io.BytesIO(read_bytes(p))


def content_sha(p, CK=None):
    """cache identity: the file's content sha (CK = mkm_contentkey) for a disk file; for a member: archive size + mtime + member CRC32 + size (no hashing of a multi-GB archive per piece)"""
    a, m = split_path(p)
    if m is None:
        return CK.content_sha(p) if CK is not None else sha256(p)
    s = stat(p)
    return "zip|%d|%d|%08x|%d" % (os.stat(a).st_size, s.st_mtime_ns, s.crc, s.st_size)


def fromfile_struct(p, dt):
    """np.fromfile(path, dtype=dt, offset=84) for a disk file; for a member the same structured array from its bytes (binary STL as stored, ascii STL parsed)"""
    import numpy as np
    a, m = split_path(p)
    if m is None:
        return np.fromfile(a, dtype=dt, offset=84)
    data = read_bytes(p)
    n = int(np.frombuffer(data[80:84], "<u4")[0]) if len(data) >= 84 else -1
    if len(data) >= 84 and len(data) == 84 + 50 * n:
        return np.frombuffer(data, dtype=dt, offset=84, count=n).copy()
    v = parse_stl(data, p); out = np.zeros(len(v), dtype=dt); out["v"] = v
    return out


def finite_tri_count(p):
    import numpy as np
    return int(len(parse_stl(read_bytes(p), p)))


def parse_stl(data, name="stl"):
    """binary or ascii STL bytes -> float32 (n,3,3), non-finite triangles dropped. Same rules as the disk loaders."""
    import numpy as np
    n = int(np.frombuffer(data[80:84], "<u4")[0]) if len(data) >= 84 else -1
    if len(data) >= 84 and len(data) == 84 + 50 * n:
        dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        v = np.frombuffer(data, dtype=dt, offset=84, count=n)["v"]
    else:
        vals = _VERTEX.findall(data)
        if not vals or len(vals) % 3:
            raise ValueError("not a readable STL (binary size mismatch, ascii vertices %d): %s" % (len(vals), name))
        v = np.array(vals, dtype=np.float64).astype(np.float32).reshape(-1, 3, 3)
    return np.ascontiguousarray(v[np.isfinite(v.reshape(-1, 9)).all(1)])


def discover(root, exts=(".stl",)):
    """R20 item 8 (MIKE 0036 item 4): pieces discovery that lists archive members WITHOUT extracting.
    Walks `root` (read-only).  A loose file with an extension in `exts` -> dict(path, size, crc=None, archive=None).  An archive (_ARCH) is opened with list_members() only: every member with a wanted
    extension -> dict(path='<archive>::<member>', size, crc, archive=<archive>).  An archive that cannot be listed (7-Zip missing, corrupt) is returned as dict(path=<archive>, error=...), never raised,
    so one bad archive does not hide the rest.  Nothing is written, moved or deleted.  Sorted by path."""
    exts = tuple(e.lower() for e in exts); out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f); lo = f.lower()
            if lo.endswith(_ARCH):
                try:
                    for name, sz, crc in list_members(p):
                        if name.lower().endswith(exts):
                            out.append(dict(path=p + SEP + name, size=sz, crc=crc, archive=p))
                except Exception as ex:      # ArchiveError, BadZipFile, OSError
                    out.append(dict(path=p, error="%s: %s" % (type(ex).__name__, ex)))
            elif lo.endswith(exts):
                out.append(dict(path=p, size=os.path.getsize(p), crc=None, archive=None))
    return sorted(out, key=lambda d: d["path"].lower())
