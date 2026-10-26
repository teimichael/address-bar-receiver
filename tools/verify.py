"""Check release integrity, anonymous metadata, and reproduced numeric outputs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = {"outputs", ".git", ".pytest_cache", ".hypothesis", "__pycache__", "node_modules"}
PRIVATE_PATH = re.compile(r"/(?:home|Users)/[^/\s\"']+|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s\"']+")
PRIVATE_KEY = re.compile(r"^-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", re.M)
TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{24,})\b")
WITHHELD_TABLES = {"browser_use.csv", "browser_use_llm.csv"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(x in GENERATED_DIRS or x.startswith(".venv") or x.endswith(".egg-info") for x in rel.parts):
            continue
        if p.is_symlink():
            raise ValueError(f"release contains a symlink: {rel}")
        if p.is_file() and p.name != "CHECKSUMS.sha256":
            yield p


def safe_path(base: Path, relative: str) -> Path:
    p = (base / relative).resolve()
    if not p.is_relative_to(base.resolve()) or Path(relative).is_absolute():
        raise ValueError("inventory contains a path outside its root")
    return p


def integrity() -> dict:
    checksums = ROOT / "CHECKSUMS.sha256"
    if not checksums.is_file():
        raise ValueError("CHECKSUMS.sha256 is missing")
    listed = {}
    for line in checksums.read_text().splitlines():
        sha, rel = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or rel in listed:
            raise ValueError("invalid or duplicate checksum entry")
        p = safe_path(ROOT, rel)
        if not p.is_file() or digest(p.read_bytes()) != sha:
            raise ValueError(f"checksum mismatch: {rel}")
        listed[rel] = sha
    files = list(release_files())
    actual = {str(p.relative_to(ROOT)) for p in files}
    if actual != set(listed):
        raise ValueError(f"inventory mismatch: extra={sorted(actual-set(listed))}; missing={sorted(set(listed)-actual)}")
    total = sum(p.stat().st_size for p in files) + checksums.stat().st_size
    largest = max((p.stat().st_size for p in files), default=0)
    if total > 20 * 1024**2 or largest > 10 * 1024**2:
        raise ValueError("release exceeds the 20 MiB total / 10 MiB per-file budgets")
    for p in files:
        rel = p.relative_to(ROOT)
        if p.suffix in {".md", ".rst"} and rel != Path("README.md"):
            raise ValueError(f"extra documentation: {rel}")
        if "manuscript" in rel.parts or rel.parts[:2] in {("audit", "browser_use"), ("evidence", "browser_use")} or p.name == "bu_mechanism.py":
            raise ValueError(f"excluded source directory present: {rel}")
        if p.suffix in {".pdf", ".tex", ".bib", ".pem", ".key", ".pyc"}:
            raise ValueError(f"excluded source/generated/secret file: {rel}")
        data = p.read_bytes()
        if p.suffix == ".gz":
            if data[:3] != b"\x1f\x8b\x08" or data[3] != 0 or data[4:8] != b"\0" * 4:
                raise ValueError(f"non-normalized gzip header: {rel}")
            data = gzip.decompress(data)
        text = data.decode("utf-8", errors="replace")
        if PRIVATE_PATH.search(text) or PRIVATE_KEY.search(text) or TOKEN.search(text):
            raise ValueError(f"private metadata or credential pattern: {rel}")
    provenance = json.loads((ROOT / "provenance.json").read_text())
    reference_claims = json.loads((ROOT / "expected/report/claims.json").read_text())
    if "details" in reference_claims["by_environment"]["realistic"]["hypotheses"]["H4"]:
        raise ValueError("withheld mechanism details appear in the reference report")
    for item in provenance["evidence_files"]:
        if item.get("source_scientific_sha256") != item.get("packaged_scientific_sha256"):
            raise ValueError(f"scientific-content preservation failed: {item['stored']}")
        data = safe_path(ROOT, item["stored"]).read_bytes()
        if item["compression"] == "gzip":
            data = gzip.decompress(data)
        elif item["compression"] != "identity":
            raise ValueError("unknown evidence compression")
        if digest(data) != item["expanded_sha256"]:
            raise ValueError(f"evidence content mismatch: {item['stored']}")
        safe_path(ROOT / "outputs/input/full", item["expanded"])
    return {"status": "passed", "files": len(files) + 1, "total_bytes": total,
            "largest_file_bytes": largest, "evidence_files": len(provenance["evidence_files"])}


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def normalized_rows(rows: list[dict], *, claims: bool = False) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "exp_dir"}
            for row in rows if not (claims and row.get("id") == "H4")]


def numeric_outputs(out: Path) -> dict:
    failures = []
    tables = []
    for p in sorted((ROOT / "expected/report/tables").glob("*.csv")):
        if p.name in WITHHELD_TABLES:
            continue
        q = out / "report/tables" / p.name
        if not q.is_file():
            failures.append(f"missing regenerated table: {p.name}")
        elif normalized_rows(read_csv(p), claims=p.name == "claims.csv") != normalized_rows(read_csv(q), claims=p.name == "claims.csv"):
            failures.append(f"table mismatch: {p.name}")
        else:
            tables.append(p.name)
    expected = json.loads((ROOT / "expected/numbers.json").read_text())
    actual = json.loads((out / "generated/numbers.json").read_text())
    different = [k for k in sorted(set(expected) | set(actual)) if expected.get(k) != actual.get(k)]
    if different:
        failures.append(f"{len(different)} stored values differ: {different[:12]}")
    derived = []
    for p in sorted((ROOT / "expected/derived").glob("*.csv")):
        if p.name == "bu_native.csv":
            continue
        q = out / "generated/derived" / p.name
        if not q.is_file() or read_csv(p) != read_csv(q):
            failures.append(f"derived table mismatch: {p.name}")
        else:
            derived.append(p.name)
    claims = json.loads((out / "report/claims.json").read_text())
    reference_claims = json.loads((ROOT / "expected/report/claims.json").read_text())
    def scientific_claim(value):
        if isinstance(value, dict):
            return {k: scientific_claim(v) for k, v in value.items() if k not in {"audit_path", "audit_sha256"}}
        if isinstance(value, list):
            return [scientific_claim(v) for v in value]
        return value
    hypotheses = claims["by_environment"]["realistic"]["hypotheses"]
    reference_hypotheses = reference_claims["by_environment"]["realistic"]["hypotheses"]
    compared_hypotheses = [key for key in reference_hypotheses if key != "H4"]
    for key in compared_hypotheses:
        if scientific_claim(hypotheses.get(key)) != scientific_claim(reference_hypotheses[key]):
            failures.append(f"regenerated hypothesis mismatch: {key}")
    quotes_equal = json.loads((ROOT / "expected/derived/quotes.json").read_text()) == json.loads((out / "generated/derived/quotes.json").read_text())
    if not quotes_equal:
        failures.append("seeded quotation selections differ")
    withheld = claims["by_environment"]["realistic"]["hypotheses"]["H4"]["status"]
    if withheld != "not_run":
        failures.append("withheld mechanism audit was presented as regenerated")
    cross = json.loads((out / "generated/crosscheck.json").read_text())
    values = json.loads((out / "generated/verify_values.json").read_text())
    if cross["disagreements"] or cross["total_episodes"] != 23676:
        failures.append("raw-evidence cross-check or episode count failed")
    if values["independent_raw_passes"] != 145 or values["aggregate_arithmetic_passes"] != 10 or values["retained_only"] != 1:
        failures.append("printed-value verification coverage differs from the declared coverage")
    if any(e["status"] not in {"ok", "retained_only"} for e in values["results"]):
        failures.append("printed-value verification has errors")
    exposure = json.loads((out / "exposure_validation.json").read_text())
    if exposure != {"status": "passed", "raw_dumps": 312, "errors": 0}:
        failures.append("raw exposure re-scoring failed")
    result = {"status": "failed" if failures else "passed_with_disclosure_limitations",
              "regenerated_report_tables": len(tables), "regenerated_derived_tables": len(derived),
              "regenerated_hypotheses": len(compared_hypotheses), "seeded_quotes_equal": quotes_equal,
              "stored_values_equal": len(expected) - len(different),
              "raw_comparisons": sum(v["compared_values"] for v in cross["sources"].values() if v["basis"] == "released_raw_evidence"),
              "aggregate_comparisons": sum(v["compared_values"] for v in cross["sources"].values() if v["basis"] == "retained_aggregate"),
              "printed_raw_passes": values["independent_raw_passes"],
              "printed_aggregate_passes": values["aggregate_arithmetic_passes"],
              "printed_retained_only": values["retained_only"], "withheld_audit": withheld,
              "rescored_observation_dumps": exposure["raw_dumps"],
              "failures": failures}
    if failures:
        raise ValueError(json.dumps(result))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrity-only", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    try:
        result = {"integrity": integrity()}
        if not args.integrity_only:
            result["reproduction"] = numeric_outputs(args.out.resolve())
            last_run = json.loads((args.out / "validation.json").read_text())
            if last_run.get("status") != "passed_with_disclosure_limitations":
                raise ValueError("the most recent reproduction run did not complete successfully")
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
