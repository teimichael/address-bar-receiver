"""Serving plan / argv / identity logic (pure functions, no Docker) and preflight classification."""

import json

import pytest

from originblind.config import ModelSpec
from originblind.serving import vllm


def _models():
    return [
        ModelSpec(key="qwen3.5-9b", hf_repo="Qwen/Qwen3.5-9B", revision="c2022", served_name="qwen3.5-9b",
                  pipeline="both", max_model_len=16384),
        ModelSpec(key="qwen2.5-7b", hf_repo="Qwen/Qwen2.5-7B-Instruct", revision="a09a", served_name="qwen2.5-7b",
                  pipeline="text", max_model_len=16384),
        ModelSpec(key="qwen3.5-9b-think", hf_repo="Qwen/Qwen3.5-9B", revision="c2022", served_name="qwen3.5-9b",
                  pipeline="text", max_model_len=16384, thinking=True),
        ModelSpec(key="mock", hf_repo="none/mock", served_name="mock", pipeline="text"),
    ]


def test_thinking_variant_shares_the_base_weights_and_server():
    plan = vllm.serve_plan("qwen3.5-9b-think", _models())
    assert plan.weights_key == "qwen3.5-9b" and plan.model_path == "/models/qwen3.5-9b"
    assert plan.served_names == ("qwen3.5-9b",) and set(plan.members) == {"qwen3.5-9b", "qwen3.5-9b-think"}


def test_limit_mm_only_for_vision_models():
    v = vllm.vllm_argv(vllm.serve_plan("qwen3.5-9b", _models()))
    t = vllm.vllm_argv(vllm.serve_plan("qwen2.5-7b", _models()))
    assert "--limit-mm-per-prompt" in v and "--limit-mm-per-prompt" not in t
    argv = vllm.docker_run_argv(vllm.serve_plan("qwen2.5-7b", _models()))
    assert "--rm" not in argv and "--ipc=host" in argv and "ob-models:/models:ro" in argv
    assert argv[argv.index("--model") + 1] == "/models/qwen2.5-7b"


def test_scripted_stand_ins_are_never_served():
    with pytest.raises(ValueError):
        vllm.serve_plan("mock", _models())


def test_identity_check():
    plan = vllm.serve_plan("qwen3.5-9b", _models())
    good = {"data": [{"id": "qwen3.5-9b", "root": "/models/qwen3.5-9b", "max_model_len": 16384}]}
    assert vllm.check_identity(good, plan)["ok"]
    wrong_root = {"data": [{"id": "qwen3.5-9b", "root": "/models/qwen3.5-4b", "max_model_len": 16384}]}
    assert not vllm.check_identity(wrong_root, plan)["ok"]
    missing = {"data": [{"id": "other", "root": "/models/qwen3.5-9b"}]}
    assert not vllm.check_identity(missing, plan)["ok"]


def test_weights_status_requires_the_pinned_revision():
    plan = vllm.serve_plan("qwen3.5-9b", _models())
    assert not vllm.weights_status(None, plan)[0]
    assert not vllm.weights_status({"repo": "Qwen/Qwen3.5-9B", "revision": "other"}, plan)[0]
    assert vllm.weights_status({"repo": "Qwen/Qwen3.5-9B", "revision": "c2022", "files": {}}, plan)[0]


def test_schedule_groups_models_by_weights(tmp_path):
    s = vllm.build_schedule(["configs/e1_default.yaml", "configs/e6_thinking.yaml"], namespace="t",
                            results=str(tmp_path))
    keys = [g["weights_key"] for g in s["groups"]]
    assert keys.count("qwen3.5-9b") == 1
    g9 = next(g for g in s["groups"] if g["weights_key"] == "qwen3.5-9b")
    assert {p["model"] for p in g9["pairs"]} == {"qwen3.5-9b", "qwen3.5-9b-think"}
    from originblind.config import load_spec
    from originblind.run.plan import ordered

    planned = sum(len(ordered(load_spec(c))) for c in ["configs/e1_default.yaml", "configs/e6_thinking.yaml"])
    assert s["totals"]["planned"] == planned and s["totals"]["missing"] == planned
    tsv = vllm.schedule_tsv(s)
    assert tsv.startswith("G\t") and "\nP\t" in tsv


def test_preflight_smoke_mode_skips_gpu_checks(tmp_path, monkeypatch):
    from originblind import preflight

    res = preflight.check(__import__("pathlib").Path.cwd(), ["configs/smoke.yaml"], smoke_only=True)
    for code in ("B03_docker_gpu", "B08_vllm_image", "B09_weights", "B11_vram", "B12_port"):
        assert res["blockers"][code]["status"] == "ok" and "skipped" in res["blockers"][code]["detail"]
    assert res["blockers"]["B07_configs"]["status"] == "ok"
    json.dumps(res)  # serialisable


def test_preflight_rejects_an_invalid_config(tmp_path):
    from originblind import preflight

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\nmodels: []\norigin_classes: [exact]\nflows: [login]\nsecrets: [otp]\nreplicates: 1\n")
    res = preflight.check(__import__("pathlib").Path.cwd(), [str(bad)], smoke_only=True)
    assert res["blockers"]["B07_configs"]["status"] == "BLOCKED"


def test_full_preflight_blocks_an_unfrozen_preregistration(tmp_path):
    """The confirmatory run starts only from a frozen design (B07b); a pilot may use a draft."""
    import shutil

    import yaml

    from originblind import preflight

    shutil.copytree("configs", tmp_path / "configs")
    p = tmp_path / "configs" / "prereg.yaml"
    pr = yaml.safe_load(p.read_text())
    for frozen in (False, True):
        pr["frozen"] = frozen
        p.write_text(yaml.safe_dump(pr))
        full = preflight.check(tmp_path, ["configs/smoke.yaml"], smoke_only=True, full=True)
        pilot = preflight.check(tmp_path, ["configs/smoke.yaml"], smoke_only=True, full=False)
        assert full["blockers"]["B07b_prereg"]["status"] == ("ok" if frozen else "BLOCKED")
        assert pilot["blockers"]["B07b_prereg"]["status"] == "ok"


# ---------------------------------------------------------------- host safety and namespace codes (README.md)

def _host(**kw):
    h = {"total_phys": int(63.5 * 1024 ** 3), "available": 40e9, "pause_updates_expiry": None,
         "commit_limit": 73.77e9, "committed": 30.4e9, "pagefiles": [("C:\\pagefile.sys", 5.6e9)],
         "pagefile_auto": True,
         "drives": {"C:": {"free": 65e9, "size": 2e12}, "D:": {"free": 3.9e12, "size": 4e12}},
         "procs": [("vmmemWSL", 5.7e9)], "wslconfig": {"raw": {"memory": "48GB"}}}
    h.update(kw)
    return h


def _probe(mem_gib=47.0, **host):
    return {"host": _host(**host), "vm": {"total": int(mem_gib * 1024 ** 3)}, "ob_vllm_running": False,
            "gpu": {"used_mib": 1219.0, "free_mib": 30969.0, "total_mib": 32607.0, "display_attached": True},
            "checkpoints": {"qwen3.5-9b": 19_329_384_361}}


def _nine_b():
    return [ModelSpec(key="qwen3.5-9b", hf_repo="Qwen/Qwen3.5-9B", revision="c2022", served_name="qwen3.5-9b",
                      pipeline="both", max_model_len=16384)]


def test_commit_headroom_blocks_full_until_the_pagefile_is_enlarged(monkeypatch):
    """B27: on this host's 2026-10-15 numbers the 9B server needs ~58 GB of commit, 43 GB are left."""
    from originblind import preflight

    monkeypatch.delenv("OB_ACCEPT_HOST_RISK", raising=False)
    res = preflight.host_safety(_nine_b(), full=True, probe=_probe())["blockers"]["B27_commit_headroom"]
    assert res["status"] == "BLOCKED" and "qwen3.5-9b" in res["detail"] and "pagefile" in res["detail"]
    big = preflight.host_safety(_nine_b(), full=True, probe=_probe(commit_limit=137.8e9))["blockers"]
    assert big["B27_commit_headroom"]["status"] == "ok"


def test_host_safety_blocks_the_crash_configuration_in_full_mode(monkeypatch):
    from originblind import preflight

    monkeypatch.delenv("OB_ACCEPT_HOST_RISK", raising=False)
    res = preflight.host_safety([], full=True, probe=_probe())["blockers"]
    assert res["B15_wsl_memory"]["status"] == "BLOCKED" and "74%" in res["B15_wsl_memory"]["detail"]
    assert res["B19_windows_update"]["status"] == "BLOCKED"
    assert res["B27_commit_headroom"]["status"] == "BLOCKED"            # even an unknown server needs ~67 GB
    assert res["B16_host_memory"]["status"] == "ok" and res["B20_windows_drives"]["status"] == "ok"
    assert res["B18_display_gpu"]["status"] == "WARN" and res["B17_gpu_idle"]["status"] in ("ok", "WARN")
    pilot = preflight.host_safety([], full=False, probe=_probe())["blockers"]
    assert pilot["B15_wsl_memory"]["status"] == "WARN"                 # hazards only warn outside `full`


def test_host_safety_passes_with_the_recommended_settings(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from originblind import preflight

    monkeypatch.delenv("OB_ACCEPT_HOST_RISK", raising=False)
    paused = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    res = preflight.host_safety(_nine_b(), full=True, probe=_probe(mem_gib=31.0, pause_updates_expiry=paused,
                                                                commit_limit=137.8e9))["blockers"]
    assert all(v["status"] != "BLOCKED" for v in res.values()), res


def test_accepting_the_host_risk_downgrades_and_records_it(monkeypatch):
    from originblind import preflight

    monkeypatch.setenv("OB_ACCEPT_HOST_RISK", "1")
    hc = preflight.host_safety([], full=True, probe=_probe())
    assert hc["overrides"] == {"OB_ACCEPT_HOST_RISK": True}
    b15 = hc["blockers"]["B15_wsl_memory"]
    assert b15["status"] == "WARN" and "accepted" in b15["detail"]


def test_a_busy_gpu_blocks_full_but_a_leftover_server_is_releasable(monkeypatch):
    from originblind import preflight

    monkeypatch.delenv("OB_ACCEPT_HOST_RISK", raising=False)
    busy = _probe()
    busy["gpu"] = {"used_mib": 29000.0, "free_mib": 3000.0, "total_mib": 32607.0, "display_attached": False}
    assert preflight.host_safety([], full=True, probe=busy)["blockers"]["B17_gpu_idle"]["status"] == "BLOCKED"
    busy["ob_vllm_running"] = True
    assert preflight.host_safety([], full=True, probe=busy)["blockers"]["B17_gpu_idle"]["status"] == "ok"
    no_gpu = _probe()
    no_gpu["gpu"] = None
    assert preflight.host_safety([], full=True, probe=no_gpu)["blockers"]["B17_gpu_idle"]["status"] == "BLOCKED"


def test_namespace_codes_stop_alarm_identity_store_and_stack(tmp_path):
    from pathlib import Path

    from originblind import preflight
    from originblind.config import load_spec
    from originblind.run.plan import run_identity

    cfg = Path("configs/e5_parity.yaml")
    spec = load_spec(cfg)
    ns = tmp_path / "ns"
    out = ns / spec.name / "episodes.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"episode_id": "e1", "attempt": 0, "status": "complete"}) + "\n")
    probe = {"vllm_digest": "vllm@sha256:new", "gpu_string": "RTX, 617.14, 32607 MiB"}
    res = preflight.namespace_checks([cfg], "ns", tmp_path, full=True, probe=probe)
    assert res["B22_stop_alarm"]["status"] == "ok"
    assert res["B23_resume_identity"]["status"] == "BLOCKED"            # results without a manifest.json
    (out.parent / "manifest.json").write_text(json.dumps(run_identity(spec)))
    (out.parent / "manifests").mkdir()
    (out.parent / "manifests" / "a.json").write_text(json.dumps(
        {"docker_images": {"vllm/vllm-openai:v0.31.0": "vllm@sha256:old"}, "gpu": "RTX, 617.14, 32607 MiB"}))
    (ns / "STOP").write_text('{"reason": "critical host memory"}')
    res = preflight.namespace_checks([cfg], "ns", tmp_path, full=True, probe=probe)
    assert res["B23_resume_identity"]["status"] == "ok" and res["B24_store_integrity"]["status"] == "ok"
    assert res["B22_stop_alarm"]["status"] == "BLOCKED" and "critical host memory" in res["B22_stop_alarm"]["detail"]
    assert res["B26_serving_stack"]["status"] == "BLOCKED" and "sha256:new" in res["B26_serving_stack"]["detail"]
    out.write_bytes(out.read_bytes() + b"123\n")                          # a line that crashes the resume reader
    res = preflight.namespace_checks([cfg], "ns", tmp_path, full=False, probe=probe)
    assert res["B24_store_integrity"]["status"] == "BLOCKED" and res["B26_serving_stack"]["status"] == "WARN"


def test_smoke_mode_never_probes_the_host(monkeypatch):
    from pathlib import Path

    from originblind import preflight

    def boom(*a, **k):
        raise AssertionError("host probed in smoke mode")

    monkeypatch.setattr(preflight, "host_safety", boom)
    res = preflight.check(Path.cwd(), ["configs/smoke.yaml"], smoke_only=True, full=True, namespace="x")
    assert not any(k.startswith(("B15", "B22")) for k in res["blockers"])


# ---------------------------------------------------------------- rerun readiness after the 2026-10-15 Docker purge

def test_image_pins_accept_the_index_digest_as_id_or_repo_digest(monkeypatch):
    """B08/B08b: the containerd image store reports the pinned index digest as .Id, the classic store
    only in RepoDigests; another build of the same tag, or the digest under another repo, is not the pin."""
    from subprocess import CompletedProcess

    from originblind import preflight

    dig = "sha256:" + "c1" * 32
    for out, want in ((f"{dig} vllm/vllm-openai@{dig}", True),
                      (f"sha256:{'aa' * 32} vllm/vllm-openai@{dig}", True),
                      (f"sha256:{'bb' * 32} vllm/vllm-openai@sha256:{'dd' * 32}", False),
                      (f"sha256:{'bb' * 32} evil/vllm-openai@{dig}", False)):
        monkeypatch.setattr(preflight, "_run", lambda cmd, timeout=30, o=out: CompletedProcess(cmd, 0, o, ""))
        assert preflight._pinned("vllm/vllm-openai:v0.31.0", dig)[0] is want, out
    monkeypatch.setattr(preflight, "_run", lambda cmd, timeout=30: CompletedProcess(cmd, 1, "", "No such image"))
    assert preflight._pinned("vllm/vllm-openai:v0.31.0", dig) == (None, "missing")


def test_the_lab_base_digest_is_one_pin_everywhere():
    """The Dockerfile's FROM (pulled and tagged by scripts/00_setup.sh, checked by B08b), its label and
    scripts/lab_run.sh's provenance constant name the same base digest; the vLLM pin is group 1's."""
    import re
    from pathlib import Path

    from originblind import preflight

    img, dig = preflight.lab_base(Path.cwd())
    assert img == "mcr.microsoft.com/playwright/python:v1.63.0-noble" and re.fullmatch(r"sha256:[0-9a-f]{64}", dig)
    assert f'BASE_DIGEST="{dig}"' in Path("scripts/lab_run.sh").read_text()
    assert f'base.digest="{dig}"' in Path("docker/lab.Dockerfile").read_text()
    assert vllm.IMAGE_DIGEST == "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"


def test_lab_volume_listing_tells_empty_from_a_failed_docker_run(monkeypatch):
    from subprocess import CompletedProcess

    from originblind import preflight

    for rc, out, want in ((0, "e2s_screenshot\n__ob_listed__\n", {"e2s_screenshot"}), (0, "__ob_listed__\n", set()),
                          (125, "", None), (0, "e2s_screenshot\n", None)):
        monkeypatch.setattr(preflight, "_run", lambda cmd, timeout=30, r=rc, o=out: CompletedProcess(cmd, r, o, ""))
        assert preflight.lab_volume_experiments("full") == want


def test_b25_blocks_lab_results_that_only_the_volume_holds(tmp_path):
    """scripts/lab_run.sh resumes from the lab volume when the host lacks an experiment: after archiving an
    attempt on the host only, the volume copy would silently continue under the same namespace."""
    from pathlib import Path

    from originblind import preflight

    lab, host_cfg = Path("configs/e2s_screenshot.yaml"), Path("configs/e5_parity.yaml")
    vol = {"lab_volume": {"e2s_screenshot"}}
    res = preflight.namespace_checks([host_cfg], "ns", tmp_path, full=True, probe=vol)
    assert res["B25_lab_volume"]["status"] == "ok"                       # no lab config selected
    res = preflight.namespace_checks([host_cfg, lab], "ns", tmp_path, full=True, probe=vol)
    assert res["B25_lab_volume"]["status"] == "BLOCKED" and "e2s_screenshot" in res["B25_lab_volume"]["detail"]
    assert preflight.namespace_checks([lab], "ns", tmp_path, full=False, probe=vol)["B25_lab_volume"]["status"] == "WARN"
    assert preflight.namespace_checks([lab], "ns", tmp_path, full=True,
                                      probe={"lab_volume": set()})["B25_lab_volume"]["status"] == "ok"
    res = preflight.namespace_checks([lab], "ns", tmp_path, full=True, probe={"lab_volume": None})
    assert res["B25_lab_volume"]["status"] == "BLOCKED" and "cannot list" in res["B25_lab_volume"]["detail"]
    (tmp_path / "ns" / "e2s_screenshot").mkdir(parents=True)               # both copies: an ordinary resume
    (tmp_path / "ns" / "e2s_screenshot" / "episodes.jsonl").write_text("")
    assert preflight.namespace_checks([lab], "ns", tmp_path, full=True, probe=vol)["B25_lab_volume"]["status"] == "ok"


def test_b28_blocks_a_local_llm_server_in_full_mode(monkeypatch):
    """An idle Ollama holds no VRAM but can load a model onto the GPU vLLM uses at any time."""
    from originblind import preflight
    from originblind.ops import probes

    monkeypatch.delenv("OB_ACCEPT_HOST_RISK", raising=False)
    h = probes.parse_host_json(json.dumps({"llm": [{"name": "ollama", "ws": 97e6}, {"name": "ollama app", "ws": 92e6}]}))
    assert h["llm_servers"] == [("ollama", 97_000_000), ("ollama app", 92_000_000)]
    assert probes.parse_host_json(json.dumps({"llm": {"name": "lms", "ws": 1}}))["llm_servers"] == [("lms", 1)]
    assert probes.parse_host_json("{}")["llm_servers"] == []
    running = _probe(llm_servers=h["llm_servers"])
    res = preflight.host_safety([], full=True, probe=running)["blockers"]["B28_local_llm"]
    assert res["status"] == "BLOCKED" and "ollama" in res["detail"]
    assert preflight.host_safety([], full=False, probe=running)["blockers"]["B28_local_llm"]["status"] == "WARN"
    assert preflight.host_safety([], full=True, probe=_probe())["blockers"]["B28_local_llm"]["status"] == "ok"
    monkeypatch.setenv("OB_ACCEPT_HOST_RISK", "1")
    acc = preflight.host_safety([], full=True, probe=running)["blockers"]["B28_local_llm"]
    assert acc["status"] == "WARN" and "accepted" in acc["detail"]
