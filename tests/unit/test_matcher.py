import base64
import json
from urllib.parse import quote

from originblind.oracle.matcher import find_canary
from originblind.types import CapturedRequest

TOKEN = "ob" + "A1b2C3d4E5f6G7h8J9k2M3n4"  # 26 chars, alnum


def _req(**kw):
    base = dict(channel="server", method="POST", url="https://x/login", host="x", headers={}, body=b"")
    base.update(kw)
    return CapturedRequest(**base)


def test_raw_in_body():
    r = _req(body=f'{{"pw":"{TOKEN}"}}'.encode())
    hits = find_canary(TOKEN, r)
    assert any(h.location == "body" for h in hits)


def test_percent_encoded_query():
    r = _req(url=f"https://x/c?p={quote(TOKEN)}XYZ")
    assert find_canary(TOKEN, r)


def test_base64_value_and_basic_auth():
    # value base64'd directly
    r1 = _req(body=base64.b64encode(TOKEN.encode()))
    assert find_canary(TOKEN, r1)
    # Authorization: Basic base64("user:TOKEN") — base64 at a non-zero alignment
    cred = base64.b64encode(f"alice:{TOKEN}".encode()).decode()
    r2 = _req(headers={"Authorization": f"Basic {cred}"})
    hits = find_canary(TOKEN, r2)
    assert any(h.location == "header:authorization" for h in hits)


def test_urlsafe_base64_nopad_in_body():
    enc = base64.urlsafe_b64encode(("zz" + TOKEN).encode()).rstrip(b"=")
    r = _req(body=b"blob=" + enc)
    assert find_canary(TOKEN, r)


def test_json_unicode_escape():
    body = json.dumps({"secret": TOKEN}).encode()
    # force \u escaping of a couple chars by re-encoding ensure_ascii
    assert find_canary(TOKEN, _req(body=body))


def test_negative_prefix_and_edit():
    assert not find_canary(TOKEN, _req(body=TOKEN[:-1].encode()))  # prefix only
    edited = TOKEN[:-1] + ("Z" if TOKEN[-1] != "Z" else "Y")
    assert not find_canary(TOKEN, _req(body=edited.encode()))


def test_negative_unrelated_digits():
    assert not find_canary(TOKEN, _req(body=b"ts=1718051234&n=000000"))


def test_card_pan_with_separators():
    pan = "4700123456789010"  # digit-only canary (card)
    r = _req(body=f"card=4700 1234 5678 9010&x=1".encode())
    assert find_canary(pan, r)
    r2 = _req(body=b"card=4700-1234-5678-9010")
    assert find_canary(pan, r2)


def test_crlf_wrapped_base64():
    import base64
    enc = base64.b64encode(("zz" + TOKEN + "qq").encode()).decode()
    wrapped = "\r\n".join(enc[i:i+16] for i in range(0, len(enc), 16))
    r = _req(body=("blob=" + wrapped).encode())
    assert find_canary(TOKEN, r)
