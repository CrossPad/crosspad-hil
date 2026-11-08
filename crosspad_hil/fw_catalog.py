"""crosspad.app's device catalog as the hub and release_check read it.

GET https://crosspad.app/api/device/catalog?rev=<rev> lists published releases
per kit: {"kits":[{"kit","versions":[{"version","tag","channel",...}]}]}. A
stable board takes stable versions, a beta board beta and stable (any other
channel fails closed to stable); an entry whose channel disagrees with its tag
(fw_server.tag_channel: a '-' is beta) is dropped. The order is the board's
(fw_server.version_key); a version it cannot order is skipped. The hub does not
know a board's revision, so it takes the newest version both revisions list.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections.abc import Callable

from crosspad_hil.fw_server import tag_channel, version_key

CATALOG_URL = "https://crosspad.app/api/device/catalog?rev={rev}"
REVS = ("v1", "v2")
CACHE_S = 600.0     # the Worker's own index TTL
RETRY_S = 30.0      # after a fetch that failed while no catalog is held yet
TIMEOUT_S = 5
MAX_TAG = 23        # release tags, with the leading "v"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?")


def fetch(rev: str, opener: Callable = urllib.request.urlopen) -> dict:
    req = urllib.request.Request(CATALOG_URL.format(rev=rev),
                                 headers={"User-Agent": "crosspad-hil"})
    with opener(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def _check(catalog: object) -> dict:
    """The catalog if it has the shape read here, else ValueError (the fetch is rejected whole)."""
    if not isinstance(catalog, dict) or not isinstance(catalog.get("kits"), list):
        raise ValueError("catalog: no kits list")
    for k in catalog["kits"]:
        if not isinstance(k, dict) or not isinstance(k.get("versions"), list):
            raise ValueError("catalog: kit without a versions list")
        for v in k["versions"]:
            if not isinstance(v, dict) or not isinstance(v.get("version"), str) \
                    or not isinstance(v.get("channel"), str):
                raise ValueError("catalog: malformed version entry")
    return catalog


def _versions(catalog: dict, kit: str, allowed: set[str]) -> set[str]:
    for k in catalog.get("kits", []):
        if k.get("kit") == kit:
            return {v["version"] for v in k.get("versions", [])
                    if v.get("channel") in allowed and _sendable(v["version"])
                    and v["channel"] == tag_channel(_tag(v))}
    return set()


def _tag(entry: dict) -> str:
    """The entry's tag; the version with its 'v' when the catalog omits it."""
    tag = entry.get("tag")
    return tag if isinstance(tag, str) else "v" + entry["version"]


def order_key(version: str) -> tuple | None:
    """version_key, or None for a string it cannot order (e.g. an identifier like `rc1a`)."""
    try:
        return version_key(version)
    except Exception:  # noqa: BLE001 - version_key's grammar is narrower than _sendable's
        return None


def _sendable(version: str) -> bool:
    """A version the hub may put on the wire as FW_LATEST: what the tag rules allow."""
    return len(version) < MAX_TAG and VERSION_RE.fullmatch(version) is not None


def newest(catalogs: dict[str, dict], kit: str, channel: str) -> str | None:
    allowed = {"stable", "beta"} if channel == "beta" else {"stable"}     # anything else: closed
    sets = [_versions(c, kit, allowed) for c in catalogs.values()]
    common = set.intersection(*sets) if sets else set()
    keyed = [(k, v) for v in common if (k := order_key(v)) is not None]
    return max(keyed, key=lambda kv: kv[0])[1] if keyed else None


class CatalogCache:
    """`latest()` never blocks on the network and never raises: it answers from the last good
    catalogs and, when they are older than `ttl`, refreshes them on a daemon thread (one at a
    time; `background=False` refreshes inline, for tests). A refresh that fails or returns a
    malformed catalog leaves the previous data in place. While no catalog is held a failed fetch is
    retried after `retry` seconds instead of `ttl`; `has_catalog` says whether one is held, which
    is what separates "nothing fetched yet" from "fetched, nothing on this channel"."""

    def __init__(self, fetch: Callable[[str], dict] = fetch,
                 clock: Callable[[], float] = time.monotonic,
                 ttl: float = CACHE_S, revs: tuple[str, ...] = REVS,
                 background: bool = True, retry: float = RETRY_S) -> None:
        self._fetch, self._clock, self._ttl, self._revs = fetch, clock, ttl, revs
        self._retry = retry
        self._background = background
        self._catalogs: dict[str, dict] = {}
        self._next: float | None = None       # when the next refresh is due
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _refresh(self, now: float) -> None:
        try:
            catalogs = {rev: _check(self._fetch(rev)) for rev in self._revs}
        except Exception:  # noqa: BLE001 - offline, HTTP, JSON or a malformed catalog: keep the old
            with self._lock:
                if not self._catalogs:
                    self._next = now + self._retry
            return
        with self._lock:
            self._catalogs = catalogs

    def _due(self, now: float) -> bool:
        return self._next is None or now >= self._next

    @property
    def has_catalog(self) -> bool:
        """True once a fetch has succeeded (whatever it lists)."""
        with self._lock:
            return bool(self._catalogs)

    def latest(self, kit: str, channel: str) -> str | None:
        try:
            now = self._clock()
            if self._due(now) and (self._thread is None or not self._thread.is_alive()):
                with self._lock:
                    self._next = now + self._ttl
                if self._background:
                    self._thread = threading.Thread(target=self._refresh, args=(now,),
                                                    daemon=True, name="fw-catalog")
                    self._thread.start()
                else:
                    self._refresh(now)
            with self._lock:
                catalogs = self._catalogs
            return newest(catalogs, kit, channel) if catalogs else None
        except Exception:  # noqa: BLE001 - FW_LATEST is a hint, never a reason to stop the hub
            return None


def reply_fields(text: str) -> dict[str, str]:
    """`NAME: k=v k=v` -> {k: v}."""
    _, _, body = text.strip().partition(":")
    out = {}
    for field in body.split():
        key, sep, value = field.partition("=")
        if sep:
            out[key] = value
    return out
