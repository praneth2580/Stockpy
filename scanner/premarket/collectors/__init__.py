"""Collectors package (lazy imports)."""

__all__ = [
    "fetch_fii_dii",
    "fetch_global_snapshot",
    "fetch_nifty_and_banknifty",
    "fetch_vix",
    "fetch_nifty_banknifty_chains",
]


def __getattr__(name: str):
    if name == "fetch_fii_dii":
        from scanner.premarket.collectors.fii_dii import fetch_fii_dii

        return fetch_fii_dii
    if name == "fetch_global_snapshot":
        from scanner.premarket.collectors.global_markets import fetch_global_snapshot

        return fetch_global_snapshot
    if name in {"fetch_nifty_and_banknifty", "fetch_vix"}:
        from scanner.premarket.collectors import indices

        return getattr(indices, name)
    if name == "fetch_nifty_banknifty_chains":
        from scanner.premarket.collectors.option_chain import fetch_nifty_banknifty_chains

        return fetch_nifty_banknifty_chains
    raise AttributeError(name)
