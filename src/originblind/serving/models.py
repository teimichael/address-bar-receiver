"""Model weights: fetch (resumable, verified), verify, and materialize into the Docker volume
that vLLM serves from.

Why not `hf download`: on this host the Xet-backed CLI stalled for hours and exited 0 with shards
still `.incomplete`. This fetcher talks to the Hub's HTTP API directly:
  - the file list and hashes come from /api/models/<repo>/tree/<revision> (pinned revision);
  - each file is downloaded with HTTP Range requests into blobs/<id>.incomplete and resumed
    across failures (bounded retries, stall timeout);
  - every file is verified before it is published into the standard HF cache layout
    (blobs/<id> + snapshots/<rev>/<path> symlink): LFS files by sha256 == LFS oid, small files
    by git blob sha1 == git oid; sizes must match.

Materialization (vLLM on Docker Desktop/WSL2 cannot reliably read bind-mounted WSL paths, and its
offline resolver rejects the Qwen3.5 shard names): the snapshot is streamed as a tar into the
named volume `ob-models` under /<key>, with shards renamed to `model-0000N-of-0000M.safetensors`
and the index's weight_map rewritten accordingly; sizes are then re-checked inside the volume and
a `.ob_manifest.json` (repo, revision, per-file size and sha256) is written next to the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

HF = "https://huggingface.co"
VOLUME = "ob-models"
BUSYBOX = "busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e"
# GNU tar (from the vLLM image, which serving needs anyway) receives the weights: shards larger
# than 8 GiB (e.g. gemma-4-12B's single 23.9 GB file) need PAX/GNU large-file headers, which
# BusyBox tar cannot parse ("invalid tar magic").
TAR_IMAGE = "vllm/vllm-openai:v0.31.0"
_SHARD_RE = re.compile(r"^model\.safetensors-(\d+-of-\d+)\.safetensors$")


def hf_cache() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def repo_dir(repo: str) -> Path:
    return hf_cache() / ("models--" + repo.replace("/", "--"))


@dataclass
class RemoteFile:
    path: str
    size: int
    git_oid: str
    sha256: str | None  # LFS files only

    @property
    def blob_id(self) -> str:
        return self.sha256 or self.git_oid


_PINNED = re.compile(r"^[0-9a-f]{40}$")


def list_files(repo: str, revision: str, *, attempts: int = 10, sleep=time.sleep) -> list[RemoteFile]:
    """The file list (paths, sizes, hashes) of repo@revision from the Hub API. A pinned revision (a
    commit sha) never changes, so its listing is cached next to the snapshot and later calls (verify,
    materialize) need no network; transient API failures (e.g. intermittent DNS) are retried."""
    cache = repo_dir(repo) / f".ob_tree_{revision}.json"
    if _PINNED.match(revision) and cache.exists():
        return [RemoteFile(**d) for d in json.loads(cache.read_text())]
    url = f"{HF}/api/models/{repo}/tree/{revision}?recursive=1"
    for i in range(1, attempts + 1):
        try:
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if i == attempts:
                raise
            sleep(min(60, 5 * i))
    out = []
    for f in r.json():
        if f.get("type") != "file":
            continue
        lfs = f.get("lfs") or {}
        out.append(RemoteFile(path=f["path"], size=int(lfs.get("size") or f.get("size") or 0),
                              git_oid=f.get("oid", ""), sha256=lfs.get("oid")))
    if _PINNED.match(revision):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([asdict(f) for f in out], indent=1))
    return out


def _hash_file(p: Path, algo: str, git_size: int | None = None) -> str:
    h = hashlib.new(algo)
    if git_size is not None:
        h.update(f"blob {git_size}\0".encode())
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_blob(p: Path, rf: RemoteFile) -> bool:
    if not p.exists() or p.stat().st_size != rf.size:
        return False
    if rf.sha256:
        return _hash_file(p, "sha256") == rf.sha256
    return _hash_file(p, "sha1", git_size=rf.size) == rf.git_oid


def _download(repo: str, revision: str, rf: RemoteFile, dest: Path, *, retries: int = 60,
              stall_s: float = 30.0, log=print, sleep=time.sleep) -> None:
    """`retries` bounds CONSECUTIVE attempts that add no bytes. On a flaky link (intermittent DNS,
    dropped connections; seen here for hours) every attempt that makes progress resets it, so a slow
    download still completes, while a dead link stops after about 55 minutes of backoff. A connection
    that delivers nothing for `stall_s` seconds is dropped and resumed by HTTP Range (stalls of a
    still-open connection were the main cost on that link)."""
    url = f"{HF}/{repo}/resolve/{revision}/{rf.path}"
    tmp = dest.with_name(dest.name + ".incomplete")
    attempt = idle = 0
    while idle < retries:
        attempt += 1
        have = tmp.stat().st_size if tmp.exists() else 0
        if have > rf.size:
            tmp.unlink()
            have = 0
        if have == rf.size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        t0, err = time.time(), None
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=httpx.Timeout(60.0, read=stall_s)) as r:
                if r.status_code == 200 and have:
                    have = 0  # server ignored the range: restart
                    tmp.unlink(missing_ok=True)
                elif r.status_code not in (200, 206):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                with tmp.open("ab") as fh:
                    for chunk in r.iter_bytes(8 << 20):
                        fh.write(chunk)
        except (httpx.HTTPError, OSError) as e:
            err = e
        got = (tmp.stat().st_size if tmp.exists() else 0) - have
        idle = 0 if got > 0 else idle + 1
        if err is None:
            rate = got / max(1e-6, time.time() - t0) / 1e6
            log(f"    {rf.path}: +{got/1e9:.2f} GB at {rate:.1f} MB/s (attempt {attempt})")
        else:
            log(f"    {rf.path}: attempt {attempt} failed after +{got/1e9:.2f} GB: "
                f"{type(err).__name__}: {str(err)[:120]}")
        if err is not None or got == 0:
            sleep(min(60, 5 * max(1, idle)))
    if not tmp.exists() or tmp.stat().st_size != rf.size:
        raise RuntimeError(f"{repo}:{rf.path} incomplete after {attempt} attempts "
                           f"({retries} in a row without progress)")
    if rf.sha256:
        ok = _hash_file(tmp, "sha256") == rf.sha256
    else:
        ok = _hash_file(tmp, "sha1", git_size=rf.size) == rf.git_oid
    if not ok:
        tmp.unlink()
        raise RuntimeError(f"{repo}:{rf.path} hash mismatch after download (deleted; rerun to retry)")
    tmp.rename(dest)


def fetch(repo: str, revision: str, *, parallel: int = 4, log=print) -> dict:
    """Ensure every file of repo@revision is present and verified in the HF cache layout.

    Up to `parallel` files download at once: the Hub throttles each connection (measured here at
    6-9 MB/s per stream, ~30 MB/s over four), and every file is still hash-verified on its own."""
    files = list_files(repo, revision)
    rd = repo_dir(repo)
    blobs, snap = rd / "blobs", rd / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    (rd / "refs").mkdir(exist_ok=True)
    (rd / "refs" / "main").write_text(revision)
    # one download per blob (identical files share a blob), largest first so the longest starts now;
    # a stale .incomplete from another downloader may not be a byte prefix: the final hash check
    # catches that, deletes it and the next run starts the file from zero
    lock = threading.Lock()

    def tlog(msg: str) -> None:  # one line at a time from the download threads
        with lock:
            log(msg)

    todo: dict[str, RemoteFile] = {}
    for rf in files:
        if rf.blob_id not in todo and not _verify_blob(blobs / rf.blob_id, rf):
            todo[rf.blob_id] = rf

    def one(rf: RemoteFile) -> None:
        tlog(f"  fetching {rf.path} ({rf.size/1e9:.2f} GB)")
        _download(repo, revision, rf, blobs / rf.blob_id, log=tlog)

    errors = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futures = [(rf, ex.submit(one, rf)) for rf in sorted(todo.values(), key=lambda f: -f.size)]
        for rf, fut in futures:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 (the other downloads finish first)
                errors.append(f"{rf.path}: {e}")
    if errors:
        raise RuntimeError("; ".join(errors))
    report = {}
    for rf in files:
        blob = blobs / rf.blob_id
        link = snap / rf.path
        link.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(blob, link.parent)
        if link.is_symlink() or link.exists():
            if not (link.is_symlink() and os.readlink(link) == rel):
                link.unlink()
        if not link.exists():
            link.symlink_to(rel)
        report[rf.path] = {"size": rf.size, "sha256": rf.sha256, "git_oid": rf.git_oid}
    return {"repo": repo, "revision": revision, "files": report}


def verify(repo: str, revision: str) -> dict:
    files = list_files(repo, revision)
    snap = repo_dir(repo) / "snapshots" / revision
    bad = []
    for rf in files:
        p = snap / rf.path
        if not (p.exists() and _verify_blob(p.resolve(), rf)):
            bad.append(rf.path)
    return {"repo": repo, "revision": revision, "ok": not bad, "bad": bad, "n_files": len(files)}


def _renamed(name: str) -> str:
    m = _SHARD_RE.match(name)
    return f"model-{m.group(1)}.safetensors" if m else name


def _docker(*args: str, input_stream=None, check=True, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], stdin=input_stream, capture_output=capture, text=False, check=check)


def volume_manifest(key: str, volume: str = VOLUME, timeout: float = 180) -> dict | None:
    try:  # bounded: a wedged Docker Desktop must not hang `serve start` or preflight forever
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{volume}:/m:ro", BUSYBOX, "cat",
                            f"/m/{key}/.ob_manifest.json"], capture_output=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def materialize(repo: str, revision: str, key: str, *, volume: str = VOLUME, force: bool = False,
                log=print) -> dict:
    """Stream the verified snapshot into the named volume at /<key> with vLLM-friendly names."""
    existing = volume_manifest(key, volume)
    if existing and existing.get("revision") == revision and existing.get("repo") == repo and not force:
        log(f"  {key}: already materialized at {revision[:12]}")
        return existing
    v = verify(repo, revision)
    if not v["ok"]:
        raise RuntimeError(f"cannot materialize {repo}@{revision}: unverified files {v['bad']}")
    files = list_files(repo, revision)
    snap = repo_dir(repo) / "snapshots" / revision
    subprocess.run(["docker", "volume", "create", volume], capture_output=True, check=True)
    subprocess.run(["docker", "run", "--rm", "-v", f"{volume}:/m", BUSYBOX, "sh", "-c",
                    f"rm -rf /m/{key} && mkdir -p /m/{key}"], check=True, capture_output=True)
    proc = subprocess.Popen(["docker", "run", "-i", "--rm", "--entrypoint", "tar", "-v", f"{volume}:/m",
                             TAR_IMAGE, "-C", f"/m/{key}", "-xf", "-"], stdin=subprocess.PIPE)
    expect: dict[str, int] = {}
    manifest = {"repo": repo, "revision": revision, "key": key, "files": {}}
    with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for rf in files:
            src = (snap / rf.path).resolve()
            name = _renamed(rf.path)
            if rf.path.endswith(".safetensors.index.json"):
                idx = json.loads(src.read_text())
                idx["weight_map"] = {k: _renamed(v) for k, v in idx["weight_map"].items()}
                data = json.dumps(idx, indent=2).encode()
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                expect[name] = len(data)
            else:
                tar.add(str(src), arcname=name)
                expect[name] = rf.size
            manifest["files"][name] = {"source": rf.path, "size": expect[name], "sha256": rf.sha256}
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("tar into the volume failed")
    # re-check sizes inside the volume (dotfiles included); the manifest that marks the copy as
    # ready is written ONLY after this check passes
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{volume}:/m:ro", BUSYBOX, "sh", "-c",
                        f"cd /m/{key} && find . -maxdepth 1 -type f -exec stat -c '%s %n' {{}} +"],
                       capture_output=True, text=True, check=True)
    got = {}
    for ln in r.stdout.splitlines():
        if " " in ln:
            size, name = ln.split(" ", 1)
            got[name[2:] if name.startswith("./") else name] = int(size)
    bad = {k: (v, got.get(k)) for k, v in expect.items() if got.get(k) != v}
    if bad:
        raise RuntimeError(f"size mismatch inside volume for {key}: {bad}")
    data = json.dumps(manifest, indent=2).encode()
    w = subprocess.run(["docker", "run", "-i", "--rm", "-v", f"{volume}:/m", BUSYBOX, "sh", "-c",
                        f"cat > /m/{key}/.ob_manifest.json"], input=data, capture_output=True)
    if w.returncode != 0:
        raise RuntimeError(f"could not write the volume manifest for {key}")
    log(f"  {key}: materialized {len(expect)} files into volume {volume}")
    return manifest


def verify_volume(key: str, volume: str = VOLUME, timeout: float = 3600) -> dict:
    """Re-check the materialized copy vLLM actually reads: sha256 of every file whose manifest entry
    records one (the LFS weights), size of the others. The *.safetensors.index.json is size-checked
    only: materialize() rewrites its weight_map, while the manifest keeps the source's (LFS) sha256.
    Read-only (the volume is mounted :ro)."""
    man = volume_manifest(key, volume)
    if not man:
        return {"key": key, "ok": False, "bad": ["no readable .ob_manifest.json in the volume (or docker failed)"]}
    files = man.get("files") or {}

    def rewritten(n: str) -> bool:
        return n.endswith(".safetensors.index.json")

    hashed = sorted(n for n, v in files.items() if (v or {}).get("sha256") and not rewritten(n))
    sized = sorted(n for n in files if n not in hashed)
    bad: list[str] = []

    def run(cmd: str, names: list[str]) -> dict[str, str]:
        if not names:
            return {}
        # names travel as argv ("$@"), never through the shell string; a named container can be
        # removed when the client times out (the container would otherwise keep reading)
        name = f"ob-verify-{re.sub(r'[^a-zA-Z0-9_.-]', '-', key)}-{os.getpid()}"
        try:
            r = subprocess.run(["docker", "run", "--rm", "--name", name, "-v", f"{volume}:/m:ro", BUSYBOX, "sh",
                                "-c", f'cd "/m/$0" && {cmd} "$@"', key, *names], capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, stdin=subprocess.DEVNULL)
            bad.append(f"{cmd.split()[0]} timed out after {timeout:.0f} s")
            return {}
        out: dict[str, str] = {}
        for ln in r.stdout.splitlines():
            val, _, fname = ln.partition(" ")
            out[fname.strip().lstrip("*")] = val
        if r.returncode != 0:
            bad.append(f"{cmd} failed: {r.stderr.strip()[:200]}")
        return out

    got_sha = run("sha256sum", hashed)
    for n in hashed:
        if got_sha.get(n) != files[n]["sha256"]:
            bad.append(f"{n}: sha256 {str(got_sha.get(n))[:12]} != {files[n]['sha256'][:12]}")
    got_size = run("stat -c '%s %n'", sized)
    for n in sized:
        if str(files[n].get("size")) != got_size.get(n):
            bad.append(f"{n}: size {got_size.get(n)} != {files[n].get('size')}")
    return {"key": key, "repo": man.get("repo"), "revision": man.get("revision"), "ok": not bad,
            "sha256_checked": len(hashed), "size_checked": len(sized), "bad": bad}


def _models_from_configs(paths: list[str]) -> list[dict]:
    from originblind.config import load_yaml_spec

    seen: dict[str, dict] = {}
    for p in paths:
        for m in load_yaml_spec(p).get("models", []):
            if m.get("hf_repo", "").startswith("none/"):
                continue
            seen[m["key"]] = m
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind models")
    ap.add_argument("action", choices=["fetch", "verify", "materialize", "verify-volume", "status"])
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--repo"); ap.add_argument("--revision"); ap.add_argument("--key")
    ap.add_argument("--only", nargs="*", default=None, help="model keys to act on")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--parallel", type=int, default=4, help="files downloaded at once per repo (fetch)")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    a = ap.parse_args(argv)
    specs = ([{"key": a.key or a.repo.split("/")[-1].lower(), "hf_repo": a.repo, "revision": a.revision}]
             if a.repo else _models_from_configs(a.configs))
    if a.only:
        specs = [m for m in specs if m["key"] in a.only]
    if a.action in ("fetch", "materialize", "verify", "verify-volume"):
        # one copy of weights per (repo, revision): e.g. qwen3.5-9b-think reuses qwen3.5-9b
        uniq: dict[tuple, dict] = {}
        for m in specs:
            uniq.setdefault((m.get("hf_repo"), m.get("revision")), m)
        specs = list(uniq.values())
    rc, out = 0, []
    for m in specs:
        rev = m.get("revision")
        if not rev:
            print(f"{m['key']}: no pinned revision in config", file=sys.stderr)
            rc = 1
            continue
        try:
            if a.action == "fetch":
                print(f"[fetch] {m['hf_repo']}@{rev[:12]}")
                out.append(fetch(m["hf_repo"], rev, parallel=a.parallel))
            elif a.action == "verify":
                res = verify(m["hf_repo"], rev)
                print(json.dumps(res))
                out.append(res)
                rc |= 0 if res["ok"] else 1
            elif a.action == "materialize":
                print(f"[materialize] {m['key']} <- {m['hf_repo']}@{rev[:12]}")
                out.append(materialize(m["hf_repo"], rev, m["key"], force=a.force))
            elif a.action == "verify-volume":
                res = verify_volume(m["key"])
                if res["ok"] and (res.get("repo"), res.get("revision")) != (m["hf_repo"], rev):
                    res["ok"] = False
                    res["bad"].append(f"volume holds {res.get('repo')}@{str(res.get('revision'))[:12]}, "
                                      f"not {m['hf_repo']}@{rev[:12]}")
                print(json.dumps(res))
                out.append(res)
                rc |= 0 if res["ok"] else 1
            else:
                # models that share (repo, revision) share the first such key's weights
                wkey = next(x["key"] for x in specs if (x.get("hf_repo"), x.get("revision")) == (m["hf_repo"], rev))
                vm = volume_manifest(wkey)
                res = {"key": m["key"], "weights_key": wkey, "materialized_revision": (vm or {}).get("revision"),
                       "expected_revision": rev, "ready": bool(vm) and vm.get("revision") == rev}
                print(json.dumps(res))
                out.append(res)
                rc |= 0 if res["ready"] else 1
        except Exception as e:  # noqa: BLE001
            print(f"{m['key']}: {a.action} failed: {e}", file=sys.stderr)
            rc = 1
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(out, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
