"""Terminal conversation with the local model.

Ollama is a program that runs the model and listens on this Mac at
http://127.0.0.1:11434. This file sends the chat there and prints the reply.
"""

import json
import re
import urllib.error
import urllib.request

from friday.persona import CRISP, system_prompt
from friday.search import web_search

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# The spare Linux laptop has 8 GB of RAM. This 3B model fits there.
MODEL = "qwen2.5:3b"
# Next line is a clearer question, after a search found nothing. No name required.
RETRY = "?"
YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "please",
    "ok",
    "okay",
    "go ahead",
    "do it",
    "look it up",
    "google it",
    "google",
    "search",
}
NO_WORDS = {"no", "nope", "don't", "do not", "cancel", "never mind", "stop"}


def ask(messages: list[dict]) -> str:
    """Send the whole conversation so far and return Friday's reply."""
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 2048},
        }
    ).encode()
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    return payload["message"]["content"].strip()


def take_turn(messages: list[dict], text: str) -> str | None:
    """Append one line from Sir, refresh the clock, and return Friday's reply.

    messages[0] must already be the system prompt.
    """
    messages.append({"role": "user", "content": text})
    messages[0] = {"role": "system", "content": system_prompt()}
    try:
        reply = ask(messages)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace").strip()
        print(f"Ollama answered {error.code}.")
        if detail:
            print(detail)
        messages.pop()
        return None
    except urllib.error.URLError:
        print("Friday cannot reach Ollama at http://127.0.0.1:11434.")
        print("Start Ollama, then run this again.")
        messages.pop()
        return None
    messages.append({"role": "assistant", "content": reply})
    return reply


def _search_line(reply: str) -> str | None:
    match = re.search(r"SEARCH:\s*(.+)", reply, re.IGNORECASE)
    if not match:
        return None
    query = match.group(1).strip().splitlines()[0].strip(" .")
    return query or None


def _plain(text: str) -> str:
    return text.lower().strip(" .!?,")


def _agrees(text: str) -> bool:
    word = _plain(text)
    return word in YES_WORDS or word.startswith(("yes ", "yeah ", "sure ", "please ", "ok ", "okay "))


def _declines(text: str) -> bool:
    word = _plain(text)
    return word in NO_WORDS or word.startswith(("no ", "don't ", "do not "))


# Measured and live facts. The small model states these with confidence even when it is wrong.
_MEASURED = re.compile(
    r"\b(how (tall|high|old|far|long|much|many)|who won|who (is|was)|weather|price of|score|"
    r"what(?:'s| is) the (weather|price|score|population|height))\b",
    re.IGNORECASE,
)
_ABOUT_US = re.compile(r"\bwho (are you|am i|is friday)\b", re.IGNORECASE)
_ADMITS_UNKNOWN = re.compile(
    r"\b(don'?t know|do not know|don'?t have (any )?information|do not have (any )?information|"
    r"no information|not sure|never heard)\b",
    re.IGNORECASE,
)


def _needs_lookup(text: str) -> bool:
    if _ABOUT_US.search(text):
        return False
    return _MEASURED.search(text) is not None


def _explicit_lookup(text: str) -> str | None:
    """He already asked for a search, so there is nothing left to confirm."""
    if not re.search(r"\b(google|look up|lookup|search for|search)\b", text, re.IGNORECASE):
        return None
    query = re.sub(r"\b(google|look up|lookup|search for|search)\b", " ", text, flags=re.IGNORECASE)
    query = " ".join(query.split()).strip(" ,.?!")
    return query or None


def _from_notes(query: str, notes: str) -> str | None:
    messages = [
        {"role": "system", "content": CRISP},
        {"role": "user", "content": f"Question: {query}\n\nNotes:\n{notes}"},
    ]
    try:
        return ask(messages)
    except (urllib.error.HTTPError, urllib.error.URLError):
        print("Friday cannot reach Ollama at http://127.0.0.1:11434.")
        return None


def lookup(query: str) -> str | None:
    """Search, then ask the model for a short answer that stays inside the notes."""
    print(f"Looking up: {query}")
    try:
        notes = web_search(query)
    except OSError as error:
        print(error)
        return "I could not reach the search, Sir."
    if not notes:
        return "I could not find that, Sir. Say it once more."
    return _from_notes(query, notes)


def _missed(spoken: str | None) -> bool:
    if not spoken:
        return False
    lowered = spoken.lower()
    return "could not find" in lowered or "couldn't find" in lowered


def _remember(messages: list[dict], user_line: str, spoken: str | None) -> tuple[str | None, str | None]:
    if spoken and _missed(spoken):
        spoken = "I could not find that, Sir. Say it once more."
        messages.append({"role": "user", "content": user_line})
        messages.append({"role": "assistant", "content": spoken})
        return spoken, RETRY
    if spoken:
        messages.append({"role": "user", "content": user_line})
        messages.append({"role": "assistant", "content": spoken})
    return spoken, None


def handle_turn(messages: list[dict], text: str, pending: str | None) -> tuple[str | None, str | None]:
    """One thing Sir said.

    pending is a search he has not accepted yet.
    RETRY means the last search found nothing, and this line is the question said again.
    """
    if pending == RETRY:
        if _declines(text):
            spoken = "All right, Sir."
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": spoken})
            return spoken, None
        if _agrees(text):
            return "Say the question once more, Sir.", RETRY
        spoken = lookup(text)
        if _missed(spoken):
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": "I could not find that, Sir."})
            return "I could not find that, Sir.", None
        return _remember(messages, text, spoken)

    if pending:
        if _agrees(text):
            return _remember(messages, f"Yes, look up {pending}.", lookup(pending))
        if _declines(text):
            spoken = "All right, Sir."
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": spoken})
            return spoken, None
        pending = None

    query = _explicit_lookup(text)
    if query:
        return _remember(messages, text, lookup(query))

    if _needs_lookup(text):
        messages.append({"role": "user", "content": text})
        spoken = "I don't know that, Sir. I can look it up if you want."
        messages.append({"role": "assistant", "content": spoken})
        return spoken, text

    reply = take_turn(messages, text)
    if not reply:
        return None, None
    query = _search_line(reply)
    if not query and not _ADMITS_UNKNOWN.search(reply):
        return reply, None
    messages.pop()
    spoken = "I don't know that, Sir. I can look it up if you want."
    messages.append({"role": "assistant", "content": spoken})
    return spoken, query or text


def main() -> None:
    messages = [{"role": "system", "content": system_prompt()}]
    pending: str | None = None
    print("Friday is here. Type a message, Sir. Type exit to stop.")
    while True:
        try:
            text = input("Sir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        reply, pending = handle_turn(messages, text, pending)
        if reply:
            print(f"Friday: {reply}")


if __name__ == "__main__":
    main()
