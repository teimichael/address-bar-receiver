"""A threaded sinkhole CONNECT/HTTP proxy.

Chromium is pointed at this proxy as its sole route. It accepts CONNECT (HTTPS) and absolute
HTTP requests to ANY host:port and relays them to one fixed target (the twin TLS/HTTP
listener), regardless of the requested host. Consequences:

  - every lookalike host, IDN host, raw IP and port reaches the single twin (the twin then
    dispatches by Host/:authority);
  - TLS terminates end to end at the twin (this is a pure byte tunnel, no interception), so
    certificates stay valid and the omnibox shows the real security state;
  - it is the egress gate: the proxy only ever dials the twin, so Chromium cannot reach the
    internet.
"""

from __future__ import annotations

import socket
import threading


class SinkholeProxy:
    def __init__(self, target_host: str, target_tls_port: int, target_http_port: int) -> None:
        self.target_host = target_host
        self.target_tls_port = target_tls_port
        self.target_http_port = target_http_port
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.port = 0
        self._stop = False

    def start(self) -> int:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(128)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.port

    def _serve(self) -> None:
        while not self._stop:
            try:
                client, _ = self._srv.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        try:
            client.settimeout(30)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    client.close()
                    return
                data += chunk
            line = data.split(b"\r\n", 1)[0].decode("latin-1")
            method = line.split(" ", 1)[0].upper()
            if method == "CONNECT":
                upstream = socket.create_connection((self.target_host, self.target_tls_port), timeout=30)
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                # forward any bytes the client already sent past the CONNECT header
                tail = data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in data else b""
                if tail:
                    upstream.sendall(tail)
                self._pump(client, upstream)
            else:
                upstream = socket.create_connection((self.target_host, self.target_http_port), timeout=30)
                upstream.sendall(data)  # forward absolute-URI request as-is
                self._pump(client, upstream)
        except Exception:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _pump(a: socket.socket, b: socket.socket) -> None:
        def relay(src, dst):
            try:
                while True:
                    d = src.recv(65536)
                    if not d:
                        break
                    dst.sendall(d)
            except Exception:
                pass
            finally:
                for s in (src, dst):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass

        t = threading.Thread(target=relay, args=(a, b), daemon=True)
        t.start()
        relay(b, a)
        t.join(timeout=1)

    def stop(self) -> None:
        self._stop = True
        if self._srv:
            try:
                self._srv.close()
            except Exception:
                pass
