# The Address Bar Is Not the Receiver — WWW ’27 artifact

Anonymous code and retained evidence for *The Address Bar Is Not the Receiver: Where LLM Web Agents Release Users’ Secrets*. The release contains 23,676 final measurement episodes and 312 framework observations。

The browser-use secret-scoping reproduction and detailed native-agent records are withheld under the paper’s disclosure policy until a fix is released or 90 days pass. The published aggregate summaries remain available.

## Contents

| Directory | Contents |
|---|---|
| `src/`, `configs/` | Harness, oracle, defenses, analysis, serving code, fixed configurations and seeds |
| `evidence/` | Compressed final episodes, probes, transcripts, observation dumps, sanitized scientific provenance |
| `expected/` | Numeric comparison fixtures and explicitly retained browser-use summaries |
| `analysis/`, `tools/` | Independent derivations, table/figure generators, release validation |
| `audit/exposure/` | Framework observation capture and scoring; exact isolated-environment requirements |
| `scripts/`, `docker/`, `tests/` | Fresh-run tools, browser lab and applicable tests |
| `provenance.json`, `CHECKSUMS.sha256` | Evidence inventory, transformations, historical fingerprints and release integrity |
| `outputs/` | Generated locally; excluded from the release and Git |


## Install and reproduce retained results

Use Linux, Python 3.12 and Poetry 2.4.1. Installation requires access to the
packages in `poetry.lock`; the subsequent retained-data analysis needs no model
endpoint, GPU, framework environment or network. Full figure rendering also
needs `pdflatex`, the LaTeX `libertine` package, and its fonts; on Debian/Ubuntu
these are supplied by `texlive-latex-extra`, `texlive-fonts-extra` and their dependencies.

```bash
poetry sync
./verify.sh --integrity-only
./reproduce.sh --data-only       # tables, values, independent checks
./reproduce.sh                   # also render the two empirical figures
./verify.sh
```

Outputs go to `outputs/`; both scripts accept `--out PATH`. `OB_PY` selects an
existing Python interpreter instead of Poetry. Scripts force imports from this
release. Evidence is expanded into `outputs/input/full`, never modified in place.
Allow approximately 200 MiB for expanded evidence and generated analysis.

`outputs/validation.json` summarizes comparisons. The generators reproduce
21,791 stored numeric strings, including summaries derived from retained
aggregates. Of the manuscript’s 156 printed value keys, 145 are checked using
released raw evidence, 10 by arithmetic over retained aggregates, and one is a
retained test-count value. Equality with a retained value is not independent
validation of its underlying experiment. Machine-readable provenance marks
every numeric key and table accordingly.

The core report leaves its preregistration hypothesis `H4` as `not_run` because
that identifier denotes the withheld browser-use audit. In the paper’s numbering
this is H2; the paper’s pushed-verdict H4 corresponds to preregistration H3 and
is regenerated normally.

## Correspondence to the paper

| Evidence group | Episodes | Reconstructed results |
|---|---:|---|
| `e1_default` | 4,536 | Calibration, flow comparisons, model competence |
| `e2_information` | 4,032 | Figure 2a, Table 3, judgment/action, IDN, probe and observation-hash analyses |
| `e2s_screenshot` | 3,024 | Omnibox, fake-bar and screenshot comparisons |
| `e3_enforcement` | 9,408 | Figure 2b, Figure 3, defense and override analyses |
| `e3_utility` | 1,680 | Declared/undeclared payment delegation |
| `e5_parity` | 756 | Browser/text-driver comparison |
| `e6_thinking` | 240 | Thinking-mode comparison |
| Exposure observations | 312 dumps | Table 2; observations are re-scored from raw text |
| Retained browser-use summaries | 84 deterministic + 84 native episodes | Secret-scoping and appendix tables; aggregate reconstruction only |

Generated empirical figures are `outputs/figures/fig_binding_plane.pdf` and
`fig_defense_heatmap.pdf`. The conceptual overview is not an experimental result.
Appendix tables and seeded quotations are generated under `outputs/generated`.

## Tests and fresh measurements

```bash
poetry run python -m playwright install chromium
poetry run pytest
OB_RESULTS="$PWD/outputs/controls" scripts/15_smoke.sh
scripts/run_fresh.sh --dry-run
scripts/run_fresh.sh
```

Tests declare browser, Docker and GPU requirements; unavailable components are
reported as skips. The scripted control asserts expected oracle/gate behavior
using an always-fill backend and does not reproduce model behavior.

Fresh full measurements require Docker, NVIDIA support, a GPU with approximately
32 GB VRAM, and roughly 130 GB for the rostered weights plus download-cache space.
Fetch external weights from the repositories and immutable revisions in
`configs/roster.yaml`, subject to their access terms. Browser binaries, container
images and operating-system packages also require external downloads.
The wrapper preserves the seven full configurations, sampling settings and
seeds; it uses separate fresh outputs and skips the withheld audit. The exposure
audit can be run separately with `audit/exposure/run.sh`; its setup uses recorded
Python versions and `npm ci` and requires Python 3.12, `uv`, Node.js and npm.
The full-run resource guards retain the recorded Windows/WSL2 host-memory probe;
a host without that probe cannot pass the default full-run preflight.

Historical fingerprints identify the evidence-producing code; cleaned release
code has its own checksum identity. Archive runs honestly record unavailable Git
identity. A fresh stochastic model run is not expected to reproduce every
episode or numerical result exactly. Exact package versions and image digests
are retained; if historical operating-system packages are unavailable, setup
fails rather than silently substituting versions.
The lab uses the [Ubuntu snapshot service](https://snapshot.ubuntu.com/) at
`20261016T030740Z`, matching the retained environment record, and checks all
523 installed package versions against `docker/dpkg-packages.lock`.

## Attribution and integrity

Project metadata declares MIT. Third-party source trees and model weights are not bundled.
Upstream identities and version/revision identifiers are preserved for
Playwright/Chromium, BrowserGym, AgentLab, browser-use, Playwright MCP, vLLM,
Qwen, Gemma and Ministral, together with the remaining Python/npm dependencies.
Their respective licenses and access terms continue to apply.

`./verify.sh` checks release checksums, evidence expansion hashes, normalized gzip
headers, forbidden private metadata, size limits, and numeric reconstruction.

Release validation is recorded in `validation.json`: 461 tests passed, including
browser and Docker capture checks; one optional slow test was skipped.