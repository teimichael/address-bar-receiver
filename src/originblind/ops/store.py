"""Integrity of a result store (one experiment directory) after crashes.

A host hang loses unsynced writes (the run log of 2026-10-14 ended in NUL bytes). The writers
(run/orchestrator.py, browser/run_browser.py: measurement code, deliberately unchanged) append one
JSON line per episode, then one gzip member per transcript, without first checking that the file
ends in a newline. Policy (README.md):

  automatic  newline_terminate(): before `originblind run` appends, a JSONL store whose last byte is
             not a newline gets one. Torn bytes stay as one corrupt line (validate and the report
             still count it), and the next record can no longer be glued onto them and lost.
  check      check_store(): read-only. Corrupt lines are tolerated by every reader (skipped, counted).
             What the readers cannot survive needs a repair: a non-UTF-8 byte (the resume reader
             raises), a JSON value that is not an object (the attempt counter raises), a corrupt gzip
             member (the transcript reader raises zlib.error and loses every later member), and a
             transcript without an episode record (on resume the episode reruns under the same
             attempt index and the first transcript would win).
  repair     repair_store(): manual (never automatic for a confirmatory namespace). Moves exactly
             those bytes into <file>.quarantine-<utc>.bin, byte for byte, with offsets and sha256s
             in a report; keeps the original as <file>.pre-repair-<utc>; writes tmp+fsync+rename.
  prefix     prefix_check(host, candidate): the lab exports its volume copy over the host copy only
             if the host copy is a byte-identical prefix of it (no host record can be lost).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

JSONL_FILES = ("episodes.jsonl", "probes.jsonl")
TRANSCRIPTS = "transcripts.jsonl.gz"
GZIP_MAGIC = b"\x1f\x8b\x08"
_CHUNK = 1 << 16


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- JSONL

def split_lines(data: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """(offset, line) for every newline-terminated line, plus the unterminated tail (b"" if none)."""
    out, pos = [], 0
    while True:
        i = data.find(b"\n", pos)
        if i < 0:
            return out, data[pos:]
        out.append((pos, data[pos:i]))
        pos = i + 1


def classify_line(line: bytes) -> str:
    """ok | blank | corrupt (tolerated, counted) | non_utf8 | non_object (both crash a reader).
    Mirrors orchestrator.read_records: str.splitlines() of the decoded text, json.loads per piece."""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return "non_utf8"
    pieces = [p for p in text.splitlines() if p.strip()]
    if not pieces:
        return "blank"
    kind = "ok"
    for p in pieces:
        try:
            v = json.loads(p)
        except json.JSONDecodeError:
            kind = "corrupt"
            continue
        if not isinstance(v, dict):
            return "non_object"
    return kind


def check_jsonl(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "status": "ok"}
    data = path.read_bytes()
    lines, tail = split_lines(data)
    counts = {"ok": 0, "blank": 0, "corrupt": 0, "non_utf8": 0, "non_object": 0}
    bad = []
    for off, ln in lines + ([(len(data) - len(tail), tail)] if tail else []):
        k = classify_line(ln)
        counts[k] += 1
        if k in ("non_utf8", "non_object"):
            bad.append({"offset": off, "length": len(ln), "kind": k})
    status = "repair" if bad else ("newline" if tail else "ok")
    return {"exists": True, "bytes": len(data), "lines": len(lines) + (1 if tail else 0), **counts,
            "nul_bytes": data.count(b"\x00"), "ends_with_newline": not tail, "tail_bytes": len(tail),
            "dangerous": bad, "status": status}


def newline_terminate(path: Path) -> bool:
    """Append b"\\n" to a non-empty file whose last byte is not a newline (fsynced). True if changed."""
    try:
        with open(path, "rb+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return False
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return False
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
            return True
    except FileNotFoundError:
        return False


def fix_tails(expdir: Path) -> list[str]:
    """The automatic fix for one experiment directory: newline-terminate its JSONL stores."""
    return [n for n in JSONL_FILES if newline_terminate(Path(expdir) / n)]


def _record_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    lines, tail = split_lines(path.read_bytes())
    for _, ln in lines + ([(0, tail)] if tail else []):
        if classify_line(ln) != "ok":
            continue
        try:
            r = json.loads(ln)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if r.get("episode_id"):
            try:
                keys.add((str(r["episode_id"]), int(r.get("attempt") or 0)))
            except (TypeError, ValueError):
                continue
    return keys


# ---------------------------------------------------------------- gzip members

def _member_at(mv: memoryview, pos: int) -> tuple[int, bytes] | None:
    """(end offset, decompressed bytes) of a complete gzip member starting at pos, else None."""
    n = len(mv)
    d = zlib.decompressobj(wbits=31)
    out, p = [], pos
    try:
        while not d.eof:
            if p >= n:
                return None
            chunk = mv[p:p + _CHUNK]
            out.append(d.decompress(chunk))
            p += len(chunk)
    except zlib.error:
        return None
    return p - len(d.unused_data), b"".join(out)


def scan_gzip(data: bytes) -> dict:
    """Members [(start, end, payload)], corrupt regions [(start, end)] (resynced at the next member
    header), zero padding [(start, end)] (tolerated by Python's gzip reader)."""
    mv, n, pos = memoryview(data), len(data), 0
    members, bad, padding = [], [], []
    while pos < n:
        if data[pos] == 0:
            j = pos
            while j < n and data[j] == 0:
                j += 1
            padding.append((pos, j))
            pos = j
            continue
        m = _member_at(mv, pos)
        if m is not None:
            members.append((pos, m[0], m[1]))
            pos = m[0]
            continue
        nxt = data.find(GZIP_MAGIC, pos + 1)
        while nxt != -1 and _member_at(mv, nxt) is None:
            nxt = data.find(GZIP_MAGIC, nxt + 1)
        end = n if nxt == -1 else nxt
        bad.append((pos, end))
        pos = end
    return {"members": members, "bad": bad, "padding": padding}


def _transcript_keys(payload: bytes) -> list[tuple[str, int]]:
    keys = []
    for ln in payload.decode("utf-8", errors="replace").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("episode_id"):
            try:
                keys.append((str(r["episode_id"]), int(r.get("attempt") or 0)))
            except (TypeError, ValueError):
                continue
    return keys


def check_transcripts(path: Path, record_keys: set[tuple[str, int]]) -> dict:
    if not path.exists():
        return {"exists": False, "status": "ok"}
    data = path.read_bytes()
    sc = scan_gzip(data)
    seen: set[tuple[str, int]] = set()
    orphans, dups = [], []
    for start, end, payload in sc["members"]:
        for k in _transcript_keys(payload):
            if k in seen:
                dups.append({"key": list(k), "offset": start})
            elif k not in record_keys:
                orphans.append({"key": list(k), "offset": start})
            seen.add(k)
    status = "repair" if (sc["bad"] or orphans or dups) else "ok"
    return {"exists": True, "bytes": len(data), "members": len(sc["members"]),
            "corrupt_regions": [list(b) for b in sc["bad"]], "zero_padding": [list(p) for p in sc["padding"]],
            "orphans": orphans, "duplicates": dups, "status": status}


# ---------------------------------------------------------------- store-level check

def check_store(expdir: Path) -> dict:
    expdir = Path(expdir)
    files = {n: check_jsonl(expdir / n) for n in JSONL_FILES}
    files[TRANSCRIPTS] = check_transcripts(expdir / TRANSCRIPTS, _record_keys(expdir / "episodes.jsonl"))
    man = expdir / "manifest.json"
    man_status = "missing"
    if man.exists():
        try:
            man_status = "ok" if isinstance(json.loads(man.read_text()), dict) else "corrupt"
        except (json.JSONDecodeError, UnicodeDecodeError):
            man_status = "corrupt"
    files["manifest.json"] = {"status": "ok" if man_status in ("ok", "missing") else "repair",
                              "manifest": man_status}
    statuses = {f["status"] for f in files.values()}
    overall = "repair" if "repair" in statuses else ("newline" if "newline" in statuses else "ok")
    return {"dir": str(expdir), "status": overall, "files": files}


def check_namespace(nsdir: Path) -> dict:
    """Every experiment directory (one holding an episodes.jsonl) under a namespace."""
    nsdir = Path(nsdir)
    exps = sorted(p.parent for p in nsdir.glob("*/episodes.jsonl")) if nsdir.is_dir() else []
    res = {p.name: check_store(p) for p in exps}
    statuses = {r["status"] for r in res.values()}
    overall = "repair" if "repair" in statuses else ("newline" if "newline" in statuses else "ok")
    return {"namespace": str(nsdir), "status": overall, "experiments": res}


# ---------------------------------------------------------------- manual repair

def _open_by_other(path: Path) -> bool:
    """True if another process has `path` open (scan of /proc/*/fd)."""
    target = str(path.resolve())
    me = str(os.getpid())
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or proc.name == me:
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) == target:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".repair-tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _quarantine(path: Path, segments: list[tuple[int, bytes, str]], stamp: str) -> dict:
    q = path.with_name(f"{path.name}.quarantine-{stamp}.bin")
    with open(q, "wb") as fh:
        for _, b, _ in segments:
            fh.write(b)
    return {"file": str(q), "segments": [{"offset": o, "length": len(b), "kind": k, "sha256": _sha(b)}
                                         for o, b, k in segments]}


def repair_store(expdir: Path, *, min_age_s: float = 60.0) -> dict:
    """Quarantine what the readers cannot survive (see the module docstring). Never deletes bytes."""
    expdir = Path(expdir)
    stamp = _utc()
    rep: dict = {"dir": str(expdir), "utc": stamp, "files": {}}
    paths = [expdir / n for n in (*JSONL_FILES, TRANSCRIPTS) if (expdir / n).exists()]
    for p in paths:
        if time.time() - p.stat().st_mtime < min_age_s:
            raise RuntimeError(f"{p} changed less than {min_age_s:.0f} s ago; is a run still writing it?")
        if _open_by_other(p):
            raise RuntimeError(f"{p} is open by another process; stop the writer first")
    for name in JSONL_FILES:
        p = expdir / name
        chk = check_jsonl(p)
        if not chk["exists"] or chk["status"] == "ok":
            continue
        data = p.read_bytes()
        lines, tail = split_lines(data)
        keep, quarantine = [], []
        for off, ln in lines + ([(len(data) - len(tail), tail)] if tail else []):
            k = classify_line(ln)
            (quarantine.append((off, ln + b"\n", k)) if k in ("non_utf8", "non_object") else keep.append(ln + b"\n"))
        backup = p.with_name(f"{p.name}.pre-repair-{stamp}")
        shutil.copy2(p, backup)
        _atomic_write(p, b"".join(keep))
        rep["files"][name] = {"backup": str(backup), "kept_lines": len(keep), "newline_added": bool(tail),
                              "quarantine": _quarantine(p, quarantine, stamp) if quarantine else None}
    p = expdir / TRANSCRIPTS
    if p.exists():
        keys = _record_keys(expdir / "episodes.jsonl")
        data = p.read_bytes()
        sc = scan_gzip(data)
        keep, quarantine, seen = [], [], set()
        for start, end, payload in sc["members"]:
            ks = _transcript_keys(payload)
            reason = None
            if any(k in seen for k in ks):
                reason = "duplicate"
            elif any(k not in keys for k in ks):
                reason = "orphan"
            seen.update(ks)
            (quarantine.append((start, data[start:end], reason)) if reason else keep.append(data[start:end]))
        quarantine += [(s, data[s:e], "corrupt") for s, e in sc["bad"]]
        quarantine += [(s, data[s:e], "zero_padding") for s, e in sc["padding"]]
        if quarantine:
            backup = p.with_name(f"{p.name}.pre-repair-{stamp}")
            shutil.copy2(p, backup)
            _atomic_write(p, b"".join(keep))
            rep["files"][TRANSCRIPTS] = {"backup": str(backup), "kept_members": len(keep),
                                         "quarantine": _quarantine(p, sorted(quarantine), stamp)}
    rep["after"] = check_store(expdir)
    out = expdir / f"store_repair_{stamp}.json"
    out.write_text(json.dumps(rep, indent=2))
    rep["report"] = str(out)
    return rep


# ---------------------------------------------------------------- lab volume <-> host

def _prefix_problem(name: str, host: bytes, cand: bytes) -> str | None:
    """None iff `host` is a byte-identical prefix of `cand`. An unterminated host tail may be a prefix
    of the candidate's corresponding line; a NUL-only tail (a torn crash write) is ignored."""
    cut = host.rfind(b"\n") + 1
    head, tail = host[:cut], host[cut:]
    if not cand.startswith(head):
        lines_h = head.count(b"\n")
        common = 0
        for a, b in zip(head.split(b"\n"), cand.split(b"\n")):
            if a != b:
                break
            common += 1
        return f"{name}: host line {common + 1} of {lines_h} differs from the candidate"
    if tail and tail.strip(b"\x00") and not cand[cut:].startswith(tail):
        return f"{name}: the host's unterminated last line is not a prefix of the candidate's"
    return None


def prefix_check(host_dir: Path, cand_dir: Path) -> dict:
    """May the candidate (the lab volume copy) replace the host copy without losing host data?"""
    host_dir, cand_dir = Path(host_dir), Path(cand_dir)
    problems: list[str] = []
    for name in JSONL_FILES:
        h, c = host_dir / name, cand_dir / name
        if not h.exists():
            continue
        if not c.exists():
            problems.append(f"{name}: present on the host, missing in the candidate")
            continue
        why = _prefix_problem(name, h.read_bytes(), c.read_bytes())
        if why:
            problems.append(why)
    h, c = host_dir / TRANSCRIPTS, cand_dir / TRANSCRIPTS
    if h.exists():
        if not c.exists():
            problems.append(f"{TRANSCRIPTS}: present on the host, missing in the candidate")
        else:
            hd, cd = h.read_bytes(), c.read_bytes()
            hs, cs = scan_gzip(hd), scan_gzip(cd)
            hm = [hd[s:e] for s, e, _ in hs["members"]]
            cm = [cd[s:e] for s, e, _ in cs["members"]]
            if cm[:len(hm)] != hm:
                problems.append(f"{TRANSCRIPTS}: the host's {len(hm)} members are not a prefix of the candidate's")
            last_member_end = hs["members"][-1][1] if hs["members"] else 0
            if any(s < last_member_end for s, _ in hs["bad"]):
                problems.append(f"{TRANSCRIPTS}: the host copy has a corrupt member before its end")
    for name in ("manifest.json",):
        h, c = host_dir / name, cand_dir / name
        if h.exists() and (not c.exists() or h.read_bytes() != c.read_bytes()):
            problems.append(f"{name}: differs from the candidate")
    hmans = host_dir / "manifests"
    if hmans.is_dir():
        for f in sorted(hmans.glob("*.json")):
            c = cand_dir / "manifests" / f.name
            if not c.exists() or c.read_bytes() != f.read_bytes():
                problems.append(f"manifests/{f.name}: missing or different in the candidate")
    return {"ok": not problems, "problems": problems, "host": str(host_dir), "candidate": str(cand_dir)}
