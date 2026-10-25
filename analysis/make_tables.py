"""Emit the paper's data tables as LaTeX (booktabs) into outputs/generated/tab_*.tex.

Values come from the regenerated report and raw-log analyses. The browser-use appendix
table is reformatted from retained aggregates; its withheld raw records are not verified.
Rates are printed as percentages (integers) with n in the column header; the k/n behind every cell is in the
source CSV, and the independent cross-check compares those counts with the raw logs."""
from __future__ import annotations

import json
from collections import defaultdict

from common import (A1_E2, A1_E3, A2, DEFENSES_E3, DISPLAY, GEN, MODELS, read_derived, table, wilson)

OUT = GEN
OUTPUT_STATUS = {}


def pct(k, n) -> str:
    k, n = int(k), int(n)
    return f"{round(100 * k / n + 1e-9):d}"


def write(name: str, body: str) -> None:
    (OUT / f"tab_{name}.tex").write_text(body)
    OUTPUT_STATUS[f"tab_{name}.tex"] = {
        "mode": "aggregate" if name == "app_browseruse" else "raw",
        "status": "reformatted_retained_aggregate" if name == "app_browseruse" else "regenerated",
    }
    print("wrote", f"tab_{name}.tex")


MODEL_ROW = {m: DISPLAY[m] for m in MODELS}

# Table 2: exposure audit
expo = table("exposure")


def ex(pipe, cls, col):
    r = next(r for r in expo if r["pipeline"] == pipe and r["origin_class"] == cls)
    return r[col]


def mark(v):
    v = float(v)
    return r"\yes" if v >= 0.999 else (r"\no" if v <= 0.001 else r"\some")


PIPES = [
    ("browser-use: DOM state", "bu_default", True),
    ("browser-use: cross-origin iframes off", "bu_coi_off", False),
    ("Playwright MCP: \\texttt{browser\\_snapshot}", "pw_mcp", True),
    ("Playwright: aria snapshot", "pw_aria", True),
    ("Playwright: aria snapshot, AI mode", "pw_aria_ai", False),
    ("BrowserGym: AXTree with tab list", "bgym_axtree", True),
    ("BrowserGym: AXTree, one tab", "bgym_axtree_1tab", True),
    ("BrowserGym: AXTree + pruned HTML", "bgym_axtree_html_1tab", True),
    ("BrowserGym: pruned HTML, no AXTree", "bgym_dom", False),
]
lines = [r"\begin{tabular}{@{}l c c c c@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Receiving site} \\", r"\cmidrule(l){4-5}",
         r"Observation pipeline & Default & URL & frame & action \\", r"\midrule"]
for name, pipe, default in PIPES:
    url = min(float(ex(pipe, c, 'top_url_present')) for c in ('a2_iframe', 'a2_action'))
    lines.append(f"{name} & {r'\yes' if default else r'\no'} & {mark(url)} & "
                 f"{mark(ex(pipe, 'a2_iframe', 'sink_domain_present'))} & {mark(ex(pipe, 'a2_action', 'sink_domain_present'))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("exposure", "\n".join(lines) + "\n")

# Table 3: per-model disposition
cells = table("cells")


def e2(m, arm, classes, what="leak"):
    cs = [r for r in cells if r["experiment"] == "e2_information" and r["model"] == m and r["arm"] == arm
          and r["origin_class"] in classes]
    return sum(int(r[f"{what}_k"]) for r in cs), sum(int(r[f"{what}_n"]) for r in cs)


cpl = {r["model"]: r for r in read_derived("coupling")}
PATTERN = {}
lines = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{1}{c}{Compl.} & \multicolumn{3}{c}{Release} & \multicolumn{2}{c}{Elicited judgment} \\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(l){6-7}",
         r"Model & legit. & lookalike & iframe & action & mismatch & rel.$\mid$mism. \\",
         r" & {\footnotesize $n{=}48$} & {\footnotesize 72} & {\footnotesize 24} & {\footnotesize 24} & "
         r"{\footnotesize \%, $n{=}120$} & {\footnotesize $k/n$ (\%)} \\", r"\midrule"]
for m in MODELS:
    ek, en = e2(m, "T2", ["exact"], "r_bound")
    a1 = e2(m, "T2", A1_E2)
    ifr = e2(m, "T2", ["a2_iframe"])
    act = e2(m, "T2", ["a2_action"])
    c = cpl[m]
    jm = pct(c["dec_jm"], c["dec_n"])
    kdg = f"{c['kdg_k']}/{c['kdg_n']} ({pct(c['kdg_k'], c['kdg_n'])})"
    if c["valid"] != "True":
        kdg = r"\multicolumn{1}{c}{---$^\ast$}"
        jm = r"---$^\ast$"
    lines.append(f"{MODEL_ROW[m]} & {pct(ek, en)} & {pct(*a1)} & {pct(*ifr)} & {pct(*act)} & {jm} & {kdg} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("disposition", "\n".join(lines) + "\n")

# Appendix: E2 by arm
lines = [r"\begin{tabular}{@{}l l r r r r r@{}}", r"\toprule",
         r"Model & Observation & legit. & typo/combo/IDN & iframe & action & IDN only \\",
         r" & & {\footnotesize $n{=}48$} & {\footnotesize 72} & {\footnotesize 24} & {\footnotesize 24} & "
         r"{\footnotesize 24} \\", r"\midrule"]
ARMS = [("T0", "no URL"), ("T1", "URL"), ("T1u", "Unicode URL"), ("T2", "URL + receiver")]
for m in MODELS:
    for i, (arm, lab) in enumerate(ARMS):
        ek, en = e2(m, arm, ["exact"], "r_bound")
        idn = e2(m, arm, ["idn"])
        if arm == "T1u":
            row = f"{MODEL_ROW[m] if i == 0 else ''} & {lab} & {pct(ek, en)} & -- & -- & -- & {pct(*idn)} \\\\"
        else:
            row = (f"{MODEL_ROW[m] if i == 0 else ''} & {lab} & {pct(ek, en)} & {pct(*e2(m, arm, A1_E2))} & "
                   f"{pct(*e2(m, arm, ['a2_iframe']))} & {pct(*e2(m, arm, ['a2_action']))} & {pct(*idn)} \\\\")
        lines.append(row)
    if m != MODELS[-1]:
        lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_e2", "\n".join(lines) + "\n")

# Appendix: E3 abstention and tool use
by = {(r["model"], r["defense"]): r for r in table("defense")}
NAMES = {"D1": "Instruction", "D2": "Annotations", "D3": "Check tool", "D3info": "Pushed facts",
         "D3push": "Pushed verdict", "D5r": "Top-level gate", "D6r": "Receiver gate"}
SH = {"qwen3.5-2b": "Q-2B", "qwen3.5-4b": "Q-4B", "qwen3.5-9b": "Q-9B", "qwen3.5-35b-a3b": "Q-35B",
      "gemma-4-12b": "G-12B", "ministral-3-8b": "M-8B", "ministral-3-14b": "M-14B"}
lines = [r"\begin{tabular}{@{}l " + "r" * len(MODELS) + "@{}}", r"\toprule",
         "Defense & " + " & ".join(SH[m] for m in MODELS) + r" \\", r"\midrule",
         r"\multicolumn{8}{@{}l}{\emph{Abstention on deceptive pages (\%, $n{=}144$)}} \\"]
for d in DEFENSES_E3:
    lines.append(NAMES[d] + " & " + " & ".join(
        pct(by[(m, d)]["deceptive_abstained_k"], by[(m, d)]["deceptive_abstained_n"]) for m in MODELS) + r" \\")
flow = {(r["model"], r["defense"]): r for r in read_derived("outcome_flow")
        if r["experiment"] == "e3_enforcement" and r["group"] == "dec"}
lines.append(r"\midrule")
lines.append(r"\multicolumn{8}{@{}l}{\emph{Step limit reached on deceptive pages (\%, $n{=}144$)}} \\")
for d in ("D6r",):
    lines.append(NAMES[d] + " & " + " & ".join(pct(flow[(m, d)]["step_limit"], flow[(m, d)]["n"]) for m in MODELS)
                 + r" \\")
sub = {(r["model"], r["defense"], r["origin_class"]): r for r in read_derived("exposure")
       if r["experiment"] == "e3_enforcement"}
lines.append(r"\midrule")
lines.append(r"\multicolumn{8}{@{}l}{\emph{Pushed verdict, injected iframe ($k$ of 24)}} \\")
for lab, col in (("Secret substituted", "substituted"), ("Secret released", "released")):
    lines.append(lab + " & " + " & ".join(sub[(m, "D3push", "a2_iframe")][col] for m in MODELS) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{8}{@{}l}{\emph{Episodes that called the check tool (\%, $n{=}192$)}} \\")
lines.append("Check tool & " + " & ".join(pct(by[(m, 'D3')]['check_origin_any_k'], by[(m, 'D3')]['check_origin_any_n'])
                                          for m in MODELS) + r" \\")
td = {(r["model"], r["group"], r["what"]): r for r in read_derived("tool_deny")}
lines.append(r"Released after a DENY & " + " & ".join(
    (f"{td[(m, 'all', 'deny_before_fill')]['leaked']}/{td[(m, 'all', 'deny_before_fill')]['episodes']}"
     if int(td[(m, 'all', 'deny_before_fill')]['episodes']) else "--") for m in MODELS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_e3", "\n".join(lines) + "\n")

# Appendix: utility (E3-U)
ut = {(r["model"], r["flow"], r["defense"]): r for r in read_derived("utility")}
UD = [("D0", "none"), ("D1", "instr."), ("D3push", "verdict"), ("D5r", "top gate"), ("D6r", "recv. gate")]
lines = [r"\begin{tabular}{@{}l " + "r" * 10 + "@{}}", r"\toprule",
         r" & \multicolumn{5}{c}{Processor declared in CSP} & \multicolumn{5}{c}{Processor not declared} \\",
         r"\cmidrule(lr){2-6}\cmidrule(l){7-11}",
         "Model & " + " & ".join(lab for _, lab in UD) + " & " + " & ".join(lab for _, lab in UD) + r" \\",
         r"\midrule"]
for m in MODELS + ["all"]:
    name = "All (\\%)" if m == "all" else SH[m]
    vals = []
    for fl in ("psp", "psp_nocsp"):
        for d, _ in UD:
            r = ut[(m, fl, d)]
            vals.append(pct(r["success_k"], r["success_n"]) if m == "all" else r["success_k"])
    if m == "all":
        lines.append(r"\midrule")
    lines.append(name + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_utility", "\n".join(lines) + "\n")

# Appendix: text vs pixels
mod = {(r["model"], r["arm"]): r for r in read_derived("modality")}
lines = [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Text observation} & \multicolumn{3}{c}{Screenshot + element list} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Model & no URL & URL & URL + recv. & page only & window & + fake bar \\", r"\midrule"]
for m in MODELS:
    lines.append(SH[m] + " & " + " & ".join(
        f"{mod[(m, a)]['a1_k']}" for a in ("T0", "T1", "T2", "S0", "S1", "S2")) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_modality", "\n".join(lines) + "\n")

# Appendix: calibration by flow (E1)
e1 = {(r["model"], r["defense"], r["flow"]): r for r in read_derived("e1_flows")}
lines = [r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{No instruction} & \multicolumn{3}{c}{Instruction} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & login & code & card & login & code & card & injected$^\ddagger$ \\", r"\midrule"]
for m in MODELS:
    vals = [pct(e1[(m, d, fl)]["a1_k"], e1[(m, d, fl)]["a1_n"]) for d in ("D0", "D1") for fl in ("login", "otp", "checkout")]
    a2 = e1[(m, "D1", "all")]
    vals.append(pct(a2["a2_k"], a2["a2_n"]))
    lines.append(SH[m] + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_calibration", "\n".join(lines) + "\n")

# Appendix: browser-use grid
bu = [r for r in table("browser_use") if r["sensitive_data_format"] == "domain"]
CLS = [("exact", "legitimate"), ("typosquat", "typosquat"), ("combosquat", "combosquat"), ("idn", "IDN homograph"),
       ("subdomain", "subdomain"), ("a2_iframe", "injected iframe"), ("a2_action", "off-site form action")]
CFG = [("default", "unset"), ("true", "unset"), ("false", "unset"), ("default", "bound"), ("true", "bound"),
       ("false", "bound")]


def outcome(r):
    for k in ("leak", "bound", "withheld", "blocked", "no_field"):
        if int(r[k]):
            return {"leak": r"\textbf{leak}", "bound": "sent to site", "withheld": "withheld", "blocked": "blocked",
                    "no_field": "no field"}[k]
    return "none"


lines = [r"\begin{tabular}{@{}l " + "c" * len(CFG) + "@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{\texttt{allowed\_domains} unset} & \multicolumn{3}{c}{\texttt{allowed\_domains} = site} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Page & default & on & off & default & on & off \\", r"\midrule"]
for cls, lab in CLS:
    vals = []
    for coi, allowed in CFG:
        r = next(x for x in bu if x["origin_class"] == cls and x["cross_origin_iframes"] == coi
                 and x["allowed_domains"] == allowed)
        vals.append(outcome(r))
    lines.append(lab + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_browseruse", "\n".join(lines) + "\n")

# Appendix: elicited judgments (E2, URL+receiver)
cp = {r["model"]: r for r in read_derived("coupling")}
lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
         r"Model & false mismatch & mismatch & receiver named & release$\mid$mismatch & release$\mid$match \\",
         r" & {\footnotesize legit., $n{=}48$} & {\footnotesize deceptive, $n{=}120$} & {\footnotesize deceptive} & & \\",
         r"\midrule"]
for m in MODELS:
    r = cp[m]
    nojm = f"{r['nojm_k']}/{r['nojm_n']}" if int(r["nojm_n"]) else "--"
    star = "" if r["valid"] == "True" else r"$^\ast$"
    lines.append(f"{SH[m]}{star} & {r['exact_false_mismatch']}/{r['exact_n']} & {r['dec_jm']}/{r['dec_n']} & "
                 f"{r['dec_recognized']}/{r['dec_n']} & {r['kdg_k']}/{r['kdg_n']} & {nojm} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_probe", "\n".join(lines) + "\n")

# verbatim quotes (seeded draw)
import json as _json
import re as _re

from common import DERIVED as _D


def tex_escape(t: str) -> str:
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
           "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "`": ""}
    return "".join(rep.get(ch, ch) for ch in t)


quotes = _json.loads((_D / "quotes.json").read_text())
for m, items in quotes.items():
    for i, q in enumerate(items, 1):
        text = q["fill_step_output"].split("Action:")[0].strip()
        sents = _re.split(r"(?<=[.!?])\s+", text)
        body = tex_escape(" ".join(sents[:3])) + (r" [\dots]" if len(sents) > 3 else "")
        (OUT / f"quote_{SH[m].replace('-', '').lower()}_{i}.tex").write_text(body + "\n")
print("wrote quotes")

# Appendix: information experiment (compact)
lines = [r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Legitimate} & \multicolumn{3}{c}{Lookalike} & \multicolumn{2}{c}{Injected} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}",
         r"Model & \oURL & \oRecv & \oNoURL & \oURL & \oRecv & \oURL & \oRecv \\", r"\midrule"]
for m in MODELS:
    vals = [e2(m, "T1", ["exact"], "r_bound")[0], e2(m, "T2", ["exact"], "r_bound")[0],
            e2(m, "T0", A1_E2)[0], e2(m, "T1", A1_E2)[0], e2(m, "T2", A1_E2)[0],
            e2(m, "T1", A2)[0], e2(m, "T2", A2)[0]]
    lines.append(SH[m] + " & " + " & ".join(str(v) for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("app_info", "\n".join(lines) + "\n")

(OUT / "table_outputs.json").write_text(json.dumps(OUTPUT_STATUS, indent=2) + "\n")
