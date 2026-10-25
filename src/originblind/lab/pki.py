"""A local CA and one multi-SAN leaf covering the whole deterministic hostname set.

All twins share one certificate so the TLS listener needs no SNI logic. The leaf's SANs
enumerate every brand's apex plus every lookalike host (Unicode IDN labels are added in
their punycode/ASCII form, which is what the certificate layer sees) plus the raw-IP floor
as an IP SAN plus a wildcard for synthetic subdomains and PSP/SSO origins. When the CA is
trusted (NSS DB inside the lab image), the real Chromium omnibox shows a normal secure
state for every condition, removing the TLS confound.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

_EPOCH = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)  # fixed validity window (keys are random, so certs are not byte-reproducible; only the window is pinned)


def _key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def build_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = _key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "origin-blind local CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(_EPOCH)
        .not_valid_after(_EPOCH + _dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                key_encipherment=False, content_commitment=False, data_encipherment=False,
                key_agreement=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _sans(hosts: list[str]) -> list[x509.GeneralName]:
    sans: list[x509.GeneralName] = []
    for h in hosts:
        h = h.strip().rstrip(".")
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    return sans


def build_leaf(
    ca_key: rsa.RSAPrivateKey,
    ca_cert: x509.Certificate,
    hosts: list[str],
    wildcards: list[str] | None = None,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = _key()
    names = list(dict.fromkeys(hosts + (wildcards or [])))
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "origin-blind twin")]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(2)
        .not_valid_before(_EPOCH)
        .not_valid_after(_EPOCH + _dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(_sans(names)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert


def _pem_key(k: rsa.RSAPrivateKey) -> bytes:
    return k.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )


def _pem_cert(c: x509.Certificate) -> bytes:
    return c.public_bytes(serialization.Encoding.PEM)


def write_pki(out_dir: Path, hosts: list[str], wildcards: list[str] | None = None) -> dict[str, Path]:
    """Create ca.pem, twin.key, twin.crt (leaf+chain) in out_dir. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ca_key, ca_cert = build_ca()
    leaf_key, leaf_cert = build_leaf(ca_key, ca_cert, hosts, wildcards)
    paths = {
        "ca": out_dir / "ca.pem",
        "key": out_dir / "twin.key",
        "crt": out_dir / "twin.crt",
    }
    paths["ca"].write_bytes(_pem_cert(ca_cert))
    paths["key"].write_bytes(_pem_key(leaf_key))
    paths["crt"].write_bytes(_pem_cert(leaf_cert) + _pem_cert(ca_cert))  # full chain
    return paths
