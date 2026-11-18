"""Look up a short question on the web.

Three free sources, asked at the same time, none of which needs a key:
Wikipedia for products, people, and facts; Google News for anything recent;
and Bing's public RSS results, which are patchy but sometimes the only one with a price.
DuckDuckGo's HTML search answers this machine with an empty bot check.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

_HEADERS = {"User-Agent": "Friday/1.0 (personal assistant)"}


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise OSError(f"Could not reach {urllib.parse.urlparse(url).netloc}. {error.reason}") from error


def _plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return " ".join(text.split())


def _wikipedia(query: str) -> list[str]:
    found = json.loads(
        _get(
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode(
                {"action": "query", "list": "search", "srsearch": query, "srlimit": 2, "format": "json"}
            )
        )
    )["query"]["search"]
    if not found:
        return []
    pages = json.loads(
        _get(
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "exlimit": 2,
                    "redirects": 1,
                    "titles": "|".join(hit["title"] for hit in found),
                    "format": "json",
                }
            )
        )
    )["query"]["pages"]
    intros = {page["title"]: page.get("extract", "") for page in pages.values()}
    notes = []
    for hit in found:
        intro = " ".join(intros.get(hit["title"], "").split())
        notes.append(f"- Wikipedia, {hit['title']}: {_plain(hit['snippet'])}. {intro[:400]}")
    return notes


def _news(query: str) -> list[str]:
    root = ET.fromstring(
        _get(
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
        )
    )
    notes = []
    for item in root.findall("./channel/item")[:4]:
        title = _plain(item.findtext("title") or "")
        day = (item.findtext("pubDate") or "")[:16]
        if title:
            notes.append(f"- News, {day}: {title}")
    return notes


def _bing(query: str) -> list[str]:
    root = ET.fromstring(_get("https://www.bing.com/search?format=rss&q=" + urllib.parse.quote(query)))
    notes = []
    for item in root.findall("./channel/item")[:3]:
        title = _plain(item.findtext("title") or "")
        body = _plain(item.findtext("description") or "")
        if title or body:
            notes.append(f"- Web: {title}. {body[:240]}")
    return notes


def web_search(query: str) -> str:
    """Return short notes from all three sources. Empty string means nothing came back."""
    sources = (_wikipedia, _news, _bing)
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = [pool.submit(source, query) for source in sources]
    notes: list[str] = []
    errors: list[Exception] = []
    for future in futures:
        try:
            notes.extend(future.result())
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            errors.append(error)
    if not notes and errors:
        raise OSError(errors[0])
    return "\n".join(notes)
