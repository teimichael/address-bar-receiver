"""Run the twin as a real TLS+HTTP server on loopback sockets, behind the sinkhole proxy.

Used by the real-browser (Playwright) path. The twin app and its TwinState live in-process,
so channel A (the authoritative server request log) is read directly from TwinState after the
episode, exactly as in the in-process path.
"""

from __future__ import annotations

import ssl
import threading
from pathlib import Path

import uvicorn

from originblind.browser.edgeproxy import SinkholeProxy
from originblind.lab.twin import TwinState, make_app


class TwinServer:
    def __init__(self, state: TwinState, cert: Path, key: Path) -> None:
        self.state = state
        self._cert = cert
        self._key = key
        self.tls_port = 0
        self.http_port = 0
        self.proxy: SinkholeProxy | None = None
        self._servers: list[uvicorn.Server] = []
        self._threads: list[threading.Thread] = []

    def _run_uvicorn(self, ssl_ctx: ssl.SSLContext | None) -> int:
        app = make_app(self.state)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                ssl_certfile=str(self._cert) if ssl_ctx else None,
                                ssl_keyfile=str(self._key) if ssl_ctx else None)
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        self._servers.append(server)
        self._threads.append(t)
        while not server.started:
            if not t.is_alive():
                raise RuntimeError("uvicorn failed to start")
            threading.Event().wait(0.02)
        return server.servers[0].sockets[0].getsockname()[1]

    def start(self) -> "TwinServer":
        self.tls_port = self._run_uvicorn(ssl.create_default_context())
        self.http_port = self._run_uvicorn(None)
        self.proxy = SinkholeProxy("127.0.0.1", self.tls_port, self.http_port)
        self.proxy.start()
        return self

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy.port}"

    def stop(self) -> None:
        for s in self._servers:
            s.should_exit = True
        if self.proxy:
            self.proxy.stop()
