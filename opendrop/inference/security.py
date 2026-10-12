"""Local-server hardening helpers for the OpenDrop API.

The API server is meant to be reached from the user's own machine. Browsers
can still reach ``127.0.0.1`` from any web page the user visits, so the server
must not trust a request just because it arrived on loopback:

* Cross-origin requests are only allowed from loopback origins (plus any
  origins listed in ``OPENDROP_CORS_ORIGINS``). A request whose ``Origin``
  header names another site is refused before it reaches a route. This also
  covers DNS rebinding, where the page's origin is the attacker's hostname.
* ``OPENDROP_API_KEY`` (optional) turns on bearer-token auth for every route
  except ``/health`` and the web UI page. Send ``Authorization: Bearer <key>``
  or ``X-API-Key: <key>``.
* Binding to a non-loopback address without an API key is refused.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import re

API_KEY_ENV = "OPENDROP_API_KEY"
CORS_ORIGINS_ENV = "OPENDROP_CORS_ORIGINS"

# http(s)://localhost, 127.x.x.x or [::1], any port.
LOOPBACK_ORIGIN_RE = re.compile(
    r"^https?://(localhost|127(?:\.\d{1,3}){3}|\[::1\])(?::\d{1,5})?$",
)

# Paths reachable without the API key (liveness + the static web UI shell).
PUBLIC_PATHS = frozenset({"/health", "/", "/ui", "/favicon.ico"})


def api_key_from_env() -> str | None:
    """Return the configured API key, or None when unset/blank."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def extra_origins_from_env() -> frozenset[str]:
    """Origins allowed in addition to loopback, from a comma-separated env var."""
    raw = os.environ.get(CORS_ORIGINS_ENV, "")
    return frozenset(o.strip().rstrip("/") for o in raw.split(",") if o.strip())


def is_allowed_origin(origin: str, extra: frozenset[str] = frozenset()) -> bool:
    """True if a browser ``Origin`` may talk to the API."""
    origin = origin.strip()
    if not origin or origin == "null":
        # "null" is sent by sandboxed iframes and file:// pages: never trust it.
        return False
    if LOOPBACK_ORIGIN_RE.match(origin):
        return True
    return origin.rstrip("/") in extra


def is_loopback_host(host: str) -> bool:
    """True for addresses that only accept connections from this machine."""
    h = host.strip().strip("[]")
    if h.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def presented_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Extract the client's key from ``Authorization: Bearer`` or ``X-API-Key``."""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return None


def key_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison of the presented key against the expected one."""
    if presented is None:
        return False
    return hmac.compare_digest(presented.encode(), expected.encode())


def check_bind_allowed(host: str, api_key: str | None) -> None:
    """Refuse to expose an unauthenticated server beyond loopback."""
    if not is_loopback_host(host) and not api_key:
        raise ValueError(
            f"Refusing to bind to non-loopback address {host!r} without an API key. "
            f"Set {API_KEY_ENV} to require authentication, or bind to 127.0.0.1."
        )
