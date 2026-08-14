"""Permalinks included in ToO posts.

None of the sibling repositories contained a Cerro Pachon weather link, so the
defaults in `chatterbox.config.LinksConfig` are a starting point to be edited
rather than authoritative endpoints. Everything here is config-driven so the
set can change without touching code.
"""

import logging
from urllib.parse import quote

from .config import LinksConfig

__all__ = ["site_links", "gracedb_link", "format_link_list"]

logger = logging.getLogger(__name__)


def site_links(config: LinksConfig) -> list[tuple[str, str]]:
    """Ordered ``(label, url)`` pairs for the site conditions section.

    Parameters
    ----------
    config : `LinksConfig`
        Link configuration.

    Returns
    -------
    links : `list` [`tuple`]
        Only entries with a non-empty URL.
    """
    candidates = [
        ("Weather at Pachon", config.weather),
        ("Seeing / webcams", config.seeing),
        ("Almanac", config.almanac),
        ("Observatory status", config.observatory_status),
    ]
    candidates.extend(config.extra.items())
    return [(label, url) for label, url in candidates if url]


def gracedb_link(source: str) -> str:
    """GraceDB superevent page for a source id."""
    return f"https://gracedb.ligo.org/superevents/{quote(source)}/view/"


def format_link_list(links: list[tuple[str, str]], separator: str = "  |  ") -> str:
    """Render links as Slack mrkdwn ``<url|label>`` entries."""
    return separator.join(f"<{url}|{label}>" for label, url in links)
