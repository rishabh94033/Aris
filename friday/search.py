"""Look up a short question on the web.

This uses DuckDuckGo, not the Google API. A personal assistant can reach it
without an API key. The company lock on Microsoft apps does not apply here.
"""

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException


def web_search(query: str) -> str:
    """Return a few short notes. Empty string means nothing useful came back."""
    rows: list[dict] = []
    error: Exception | None = None
    for _attempt in range(2):
        try:
            rows = DDGS().text(query, max_results=4)
            error = None
            break
        except DDGSException as caught:
            if "no results" in str(caught).lower():
                return ""
            error = caught
            time.sleep(0.4)
    if error is not None and not rows:
        raise OSError(error)
    notes = []
    for row in rows:
        title = (row.get("title") or "").strip()
        body = (row.get("body") or "").strip().replace("\n", " ")
        if not title and not body:
            continue
        notes.append(f"- {title}. {body[:240]}")
    return "\n".join(notes)
