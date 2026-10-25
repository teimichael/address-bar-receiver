"""Action grammar and a robust parser.

Agents emit one action per step, optionally after brief reasoning. The canonical grammar:

  click(bid)
  fill(bid, "text")          # text may be a secret handle like {{password}} (quotes optional)
  type(bid, "text")
  press(bid, "Enter")
  goto("url")
  check_origin(bid)          # advisory tool (D3): returns the broker's inputs + verdict
  stop("message")            # task complete
  refuse("reason")           # decline to act

Parsing rule: the action is the LAST well-formed call to a known action in the reply, searched
line by line from the end, preferring a line that starts with "Action:". Calls never span lines,
so prose such as "the email field (e1) is visible" cannot swallow the real action (the October
pilot parser used a greedy multi-line pattern and did). A malformed reply is a parse failure (an
intent-to-treat outcome), never a crash.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

ACTIONS = ("click", "fill", "type", "press", "goto", "check_origin", "stop", "refuse")
_ACTIONS = set(ACTIONS)
# one call on one line; allows one level of nested parentheses inside quoted args
_CALL = re.compile(r"\b(?P<name>" + "|".join(ACTIONS) + r")\s*\((?P<args>(?:[^()\n]|\([^()\n]*\))*)\)")
_ACTION_PREFIX = re.compile(r"^\s*(?:\*\*)?(?:final\s+)?action(?:\*\*)?\s*:\s*", re.IGNORECASE)
_HANDLE = re.compile(r"\{\{\s*[a-z_:]+\s*\}\}?")  # "{{card}" too (defense/handles.py)


@dataclass(frozen=True)
class Action:
    name: str
    args: tuple
    raw: str
    ok: bool
    error: str = ""


def _parse_args(s: str) -> list:
    s = s.strip()
    if not s:
        return []
    # quote bare secret handles so they survive literal_eval: fill(e2, {{password}})
    s = _HANDLE.sub(lambda m: '"' + m.group(0) + '"' if not _quoted(s, m) else m.group(0), s)
    try:
        node = ast.parse(f"__f__({s})", mode="eval").body
        out = []
        for a in node.args:  # type: ignore[attr-defined]
            if isinstance(a, ast.Name):
                out.append(a.id)  # bare bid: click(e3)
            else:
                out.append(ast.literal_eval(a))
        return out
    except Exception:
        # fall back: split top-level commas, strip quotes
        parts, depth, cur, quote = [], 0, "", ""
        for ch in s:
            if quote:
                if ch == quote:
                    quote = ""
                cur += ch
                continue
            if ch in "\"'":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur)
        return [p.strip().strip("\"'") for p in parts]


def _quoted(s: str, m: re.Match) -> bool:
    before = s[: m.start()]
    return before.count('"') % 2 == 1 or before.count("'") % 2 == 1


def _calls_in(line: str) -> list[re.Match]:
    return list(_CALL.finditer(line))


def parse_action(text: str | None) -> Action:
    if text is None:
        return Action("", (), "", False, "empty")
    raw = text.strip()
    if not raw:
        return Action("", (), "", False, "empty")
    lines = [ln.strip().strip("`") for ln in raw.splitlines() if ln.strip()]
    # 1) an explicit "Action:" line wins (last one); 2) otherwise the last line containing a call
    for candidates in ([ln for ln in lines if _ACTION_PREFIX.match(ln)], lines):
        for ln in reversed(candidates):
            body = _ACTION_PREFIX.sub("", ln)
            for m in reversed(_calls_in(body)):
                name = m.group("name")
                try:
                    args = tuple(_parse_args(m.group("args")))
                except Exception as e:  # pragma: no cover - _parse_args falls back internally
                    return Action(name, (), m.group(0), False, f"argparse:{e}")
                return Action(name, args, m.group(0), True)
    return Action("", (), raw[:200], False, "no_action")
