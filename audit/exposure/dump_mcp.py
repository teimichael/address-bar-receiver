"""RQ1 exposure dump from the real Playwright MCP server (runs in .venv-bu, which carries the Python MCP
SDK 2.1.1 as a browser-use dependency; the server is @playwright/mcp@0.0.83, installed under
.venv-bu/pwmcp by setup_venvs.sh).

The server is started over stdio exactly as an MCP host would start it, plus network plumbing:
  node .venv-bu/pwmcp/node_modules/@playwright/mcp/cli.js --headless --isolated
       --proxy-server <sinkhole> --ignore-https-errors --executable-path <Chromium>
       --output-dir <tmp>
and for each condition the client calls `browser_navigate` {url: landing} and then `browser_snapshot`.
At 0.0.83 an implicit snapshot (after navigate/click/type) is written to a file and only linked from
the tool response ("### Snapshot\\n- [Snapshot](<output-dir>/page-*.yml)", snapshotToFile unless the
snapshot was requested explicitly); the page content reaches the model inline only through the
explicit `browser_snapshot` call, whose text content is the dump, verbatim: the "### Page" block
("- Page URL:", "- Page Title:") and the "### Snapshot" YAML built from page.ariaSnapshot({mode:"ai"}).
The navigate response is kept in the .json sidecar.
Pipeline written: pw_mcp (Playwright family, default configuration)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labclient import CLASSES, FLOWS, Lab, write_dump, write_error  # noqa: E402

REPO = HERE.parent.parent
MCP_CLI = REPO / ".venv-bu" / "pwmcp" / "node_modules" / "@playwright" / "mcp" / "cli.js"
_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def default_chrome() -> str:
    cands = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux*/chrome"))
    return str(cands[-1]) if cands else ""


async def run(args) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    lab = Lab(args.ports)
    out = Path(args.out)
    node = shutil.which("node")
    if not node or not MCP_CLI.exists():
        print(f"[dump_mcp] skipped: node={node!r}, {MCP_CLI} exists={MCP_CLI.exists()}", file=sys.stderr)
        return 3
    pkg = json.loads((MCP_CLI.parent / "package.json").read_text())
    work = Path(tempfile.mkdtemp(prefix="ob-mcp-"))
    chrome = args.chrome or default_chrome()
    argv = [str(MCP_CLI), "--headless", "--isolated", "--proxy-server", lab.proxy_url, "--ignore-https-errors",
            "--output-dir", str(work / "out")]
    if chrome:
        argv += ["--executable-path", chrome]
    params = StdioServerParameters(command=node, args=argv, cwd=str(work),
                                   env={k: v for k, v in os.environ.items() if k not in _PROXY_ENV})
    n = failed = 0
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
                server = {"name": getattr(info, "name", ""), "version": getattr(info, "version", "")}
                fv = (f"@playwright/mcp {pkg['version']}, playwright-core "
                      f"{pkg.get('dependencies', {}).get('playwright-core', '?')}, chrome {chrome or 'bundled'}")
                for flow in args.flows:
                    for oc in args.classes:
                        try:
                            ep = lab.episode(oc, flow)
                            nav = await session.call_tool("browser_navigate", {"url": ep["landing_url"]})
                            res = await session.call_tool("browser_snapshot", {})
                            if getattr(res, "is_error", getattr(res, "isError", False)):
                                raise RuntimeError(f"browser_snapshot returned an error: {_text(res)[:300]}")
                            text = _text(res)
                            m = re.search(r"^- Page URL: (\S+)", text, re.M)
                            write_dump(out, "pw_mcp", flow, oc, text, {
                                "pipeline": "pw_mcp", "framework": "playwright-mcp", "framework_family": "playwright",
                                "default_config": True,
                                "config": "Playwright MCP browser_snapshot tool response (default snapshot mode)",
                                "framework_version": fv, "mcp_server": server, "chrome_executable": chrome,
                                "episode": ep, "observed_top_url": m.group(1) if m else "",
                                "navigate_text": _text(nav),
                                "api": "MCP tools/call browser_navigate {url}; browser_snapshot {} -> "
                                       "result.content[*].text"})
                            n += 1
                        except Exception as e:  # noqa: BLE001 - recorded; never scored as "absent"
                            failed += 1
                            write_error(out, "pw_mcp", flow, oc, e)
                            print(f"[dump_mcp] {flow}/{oc} FAILED: {type(e).__name__}: {e}", file=sys.stderr,
                                  flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[dump_mcp] {n} conditions, {failed} failed -> {out}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ports", required=True)
    ap.add_argument("--out", required=True, help="results/<ns>/exposure/raw")
    ap.add_argument("--flows", nargs="+", default=list(FLOWS))
    ap.add_argument("--classes", nargs="+", default=list(CLASSES))
    ap.add_argument("--chrome", default=os.environ.get("OB_MCP_CHROME", ""),
                    help="Chromium binary (default: the newest Playwright-cached chromium-*)")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
