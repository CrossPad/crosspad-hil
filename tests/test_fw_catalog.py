from __future__ import annotations

from crosspad_hil import fw_catalog as fc
from crosspad_hil import host_report as hr


def cat(versions):
    return {"v": 1, "kits": [{"kit": "full", "versions": [{"version": v, "channel": c} for v, c in versions]}]}


def test_newest_respects_channel_and_device_order():
    cats = {"v1": cat([("1.1.0-rc10", "beta"), ("1.1.0-rc9", "beta"), ("1.0.1", "stable")]),
            "v2": cat([("1.1.0-rc10", "beta"), ("1.1.0-rc9", "beta"), ("1.0.1", "stable")])}
    assert fc.newest(cats, "full", "beta") == "1.1.0-rc10"
    assert fc.newest(cats, "full", "stable") == "1.0.1"


def test_newest_takes_a_version_both_revs_list():
    cats = {"v1": cat([("1.1.0-rc3", "beta"), ("1.1.0-rc2", "beta")]), "v2": cat([("1.1.0-rc2", "beta")])}
    assert fc.newest(cats, "full", "beta") == "1.1.0-rc2"


def test_cache_keeps_the_last_good_catalog_on_errors():
    calls = {"n": 0}
    now = {"t": 0.0}

    def fetch(rev):
        calls["n"] += 1
        if calls["n"] > 2:
            raise OSError("offline")
        return cat([("1.1.0-rc2", "beta")])
    cache = fc.CatalogCache(fetch=fetch, clock=lambda: now["t"], ttl=600.0)
    assert cache.latest("full", "beta") == "1.1.0-rc2"
    now["t"] = 601.0
    assert cache.latest("full", "beta") == "1.1.0-rc2"       # refresh failed, old value kept


def test_reply_fields():
    assert fc.reply_fields("FWUPD: state=idle running=v1.1.0-rc2 kit=full latest=1.1.0-rc3 channel=beta err=-\r\n") \
        == {"state": "idle", "running": "v1.1.0-rc2", "kit": "full", "latest": "1.1.0-rc3",
            "channel": "beta", "err": "-"}


def test_latest_for_prefers_the_override(tmp_path):
    class Cache:
        def latest(self, kit, channel):
            return "1.1.0-rc3"
    assert hr.latest_for(Cache(), "full", "beta", override=lambda: "9.9.9") == "9.9.9"
    assert hr.latest_for(Cache(), "full", "beta", override=lambda: None) == "1.1.0-rc3"


GOOD = cat([("1.1.0-rc2", "beta")])


def _cache(responses, background=False, clock=None):
    """A cache whose fetch answers from `responses` (a list, one per refresh; an Exception raises)."""
    state = {"i": 0, "t": 0.0}

    def fetch(rev):
        r = responses[min(state["i"] // 2, len(responses) - 1)]
        state["i"] += 1
        if isinstance(r, Exception):
            raise r
        return r
    c = fc.CatalogCache(fetch=fetch, clock=lambda: state["t"], ttl=10.0, background=background)
    return c, state


MALFORMED = [
    [],
    {"kits": None},
    {"kits": [{"kit": "full", "versions": None}]},
    {"kits": [{"kit": "full", "versions": [{"channel": "beta"}]}]},
    {"kits": [{"kit": "full", "versions": [{"version": 7, "channel": "beta"}]}]},
    {"kits": [{"kit": "full", "versions": [{"version": "1.1.0", "channel": None}]}]},
    {"kits": ["full"]},
]


def test_malformed_catalogs_keep_the_previous_good_result():
    import http.client
    for bad in [*MALFORMED, http.client.IncompleteRead(b"x"), ValueError("json")]:
        cache, state = _cache([GOOD, bad])
        assert cache.latest("full", "beta") == "1.1.0-rc2"
        state["t"] = 11.0
        assert cache.latest("full", "beta") == "1.1.0-rc2", bad


def test_a_malformed_first_fetch_gives_none_and_does_not_raise():
    for bad in MALFORMED:
        cache, _ = _cache([bad])
        assert cache.latest("full", "beta") is None


def test_unsendable_versions_are_dropped():
    cats = {"v1": cat([("1.0.0\nX", "stable"), ("1.0.0 --x", "stable"), ("v1.0.0", "stable"),
                       ("1.0.0-" + "a" * 30, "stable"), ("1.0.0", "stable")])}
    assert fc.newest(cats, "full", "stable") == "1.0.0"
    cats = {"v1": cat([("1.0.0\nX", "stable")])}
    assert fc.newest(cats, "full", "stable") is None


def test_tag_length_limit_counts_the_v():
    ok = "1.1.0-rc2"
    at_limit = "1.0.0-" + "a" * 16     # 22 chars, "v" makes 23
    too_long = "1.0.0-" + "a" * 17     # 23 chars, "v" makes 24
    cats = {"v1": cat([(ok, "beta"), (at_limit, "beta"), (too_long, "beta")])}
    assert fc._versions(cats["v1"], "full", {"beta"}) == {ok, at_limit}


def test_latest_does_not_block_while_the_fetch_hangs():
    import threading
    release = threading.Event()

    def fetch(rev):
        release.wait(5)
        return GOOD
    cache = fc.CatalogCache(fetch=fetch, clock=lambda: 0.0, ttl=10.0, background=True)
    try:
        assert cache.latest("full", "beta") is None       # returned at once, refresh in flight
        assert cache.latest("full", "beta") is None       # and no second refresh is started
    finally:
        release.set()
    cache._thread.join(5)
    assert cache.latest("full", "beta") == "1.1.0-rc2"


# -- has_catalog: "nothing fetched" is not "nothing on this channel" ----------------------------

def test_has_catalog_separates_never_fetched_from_nothing_on_the_channel():
    cache, _ = _cache([GOOD])                  # only an rc: no stable release
    assert cache.has_catalog is False
    assert cache.latest("full", "stable") is None
    assert cache.has_catalog is True           # fetched; the stable channel is simply empty
    assert cache.latest("full", "beta") == "1.1.0-rc2"
    failing, _ = _cache([OSError("offline")])
    assert failing.latest("full", "stable") is None
    assert failing.has_catalog is False


# -- a failed first fetch is retried soon, a held catalog keeps the ttl -------------------------

def _counting(responses, ttl=600.0, retry=30.0):
    state = {"fetches": 0, "t": 0.0}      # fetches = refreshes started (the first rev's fetch)

    def fetch(rev):
        if rev == "v1":
            state["fetches"] += 1
        r = responses[min(state["fetches"] - 1, len(responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return r
    return fc.CatalogCache(fetch=fetch, clock=lambda: state["t"], ttl=ttl, retry=retry,
                           background=False), state


def test_a_failed_first_fetch_is_retried_after_the_short_retry_not_the_ttl():
    cache, st = _counting([OSError("offline"), GOOD])
    assert cache.latest("full", "beta") is None
    first = st["fetches"]
    st["t"] = 29.0
    assert cache.latest("full", "beta") is None
    assert st["fetches"] == first                       # not yet
    st["t"] = 31.0
    assert cache.latest("full", "beta") == "1.1.0-rc2"  # asked again well before 600 s
    assert st["fetches"] > first


def test_the_short_retry_repeats_until_the_first_success_then_the_ttl_rules():
    cache, st = _counting([OSError("a"), OSError("b"), GOOD, OSError("c"), GOOD])
    cache.latest("full", "beta")
    for t in (31.0, 62.0):
        st["t"] = t
        cache.latest("full", "beta")
    assert cache.has_catalog is True
    n = st["fetches"]
    st["t"] = 62.0 + 599.0
    assert cache.latest("full", "beta") == "1.1.0-rc2"
    assert st["fetches"] == n                           # held: waits the ttl
    st["t"] = 62.0 + 601.0
    assert cache.latest("full", "beta") == "1.1.0-rc2"  # a failed refresh keeps the old data
    n2 = st["fetches"]
    assert n2 > n
    st["t"] = 62.0 + 601.0 + 100.0
    cache.latest("full", "beta")
    assert st["fetches"] == n2                          # and goes back to the ttl, not 30 s


# -- channels fail closed; the tag has the last word on a channel -------------------------------

def tagged(rows):
    return {"kits": [{"kit": "full", "versions": [
        {"version": v, "tag": t, "channel": c} for v, t, c in rows]}]}


def test_only_beta_widens_every_other_channel_is_stable():
    cats = {"v1": cat([("1.0.1", "stable"), ("1.1.0-rc3", "beta")])}
    assert fc.newest(cats, "full", "beta") == "1.1.0-rc3"
    for channel in ("stable", "nightly", "", "BETA", "beta "):
        want = "1.1.0-rc3" if channel == "beta" else "1.0.1"
        assert fc.newest(cats, "full", channel) == want, channel


def test_an_entry_whose_channel_disagrees_with_its_tag_is_dropped():
    cats = {"v1": tagged([("1.2.0-rc1", "v1.2.0-rc1", "stable"),      # a '-' is beta
                          ("1.1.0", "v1.1.0", "beta"),                # no '-' is stable
                          ("1.0.1", "v1.0.1", "stable"),
                          ("1.1.0-rc3", "v1.1.0-rc3", "beta")])}
    assert fc.newest(cats, "full", "stable") == "1.0.1"
    assert fc.newest(cats, "full", "beta") == "1.1.0-rc3"
    assert fc.newest({"v1": tagged([("1.2.0-rc1", "v1.2.0-rc1", "stable")])}, "full", "stable") is None


def test_a_catalog_without_tags_is_judged_by_the_version():
    cats = {"v1": cat([("1.2.0-rc1", "stable"), ("1.0.1", "stable")])}
    assert fc.newest(cats, "full", "stable") == "1.0.1"


# -- a version string version_key cannot order must not poison the kit --------------------------

def test_an_unorderable_catalog_version_is_skipped_not_raised():
    cats = {"v1": cat([("1.2.0-rc1a", "beta"), ("1.1.0-rc3", "beta")]),
            "v2": cat([("1.2.0-rc1a", "beta"), ("1.1.0-rc3", "beta")])}
    assert fc.newest(cats, "full", "beta") == "1.1.0-rc3"
    only = {"v1": cat([("1.2.0-rc1a", "beta")])}
    assert fc.newest(only, "full", "beta") is None
    assert fc.order_key("1.2.0-rc1a") is None and fc.order_key("1.2.0-rc1") is not None
    cache, _ = _cache([cat([("1.2.0-rc1a", "beta"), ("1.1.0-rc3", "beta")])])
    assert cache.latest("full", "beta") == "1.1.0-rc3"
