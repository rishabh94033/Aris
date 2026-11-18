"""Terminal conversation with the local model.

Ollama is a program that runs the model and listens on this Mac at
http://127.0.0.1:11434. This file sends the chat there and prints the reply.

Before answering, Friday asks the model one small question: does this need a
web search, and with what words? After searching, she asks it another: do these
results answer it, or what should she search for instead?
"""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable

from friday.persona import ANSWER, judge_prompt, route_prompt, system_prompt
from friday.search import web_search

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# The spare Linux laptop has 8 GB of RAM. This 3B model fits there.
MODEL = "qwen2.5:3b"
# Ollama unloads an idle model after 5 minutes, and loading it again is the slow first answer.
KEEP_ALIVE = "30m"
# Room for the instructions, the recent talk, and up to three sets of search results.
CONTEXT_TOKENS = 4096
# Older lines than this are left out of each request, so the instructions are never pushed out.
RECENT_MESSAGES = 12
MAX_SEARCHES = 3
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Asking first is Sir's rule, not the model's judgment. His own words can already be the yes.
_ASKED = re.compile(r"\b(search|look (?:it |that |this )?up|google|check)\b", re.IGNORECASE)

Say = Callable[[str], None]


def _request(messages: list[dict], stream: bool, options: dict) -> urllib.request.Request:
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": CONTEXT_TOKENS, **options},
    }
    return urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )


def ask(messages: list[dict], say: Say | None = None, **options: float) -> str:
    """Send messages to the model and return its words.

    With say, Ollama streams the words a few at a time, and each sentence is
    handed to say as soon as it is complete.
    """
    if say is None:
        with urllib.request.urlopen(_request(messages, False, options), timeout=180) as response:
            return json.load(response)["message"]["content"].strip()

    words: list[str] = []
    buffer = ""
    with urllib.request.urlopen(_request(messages, True, options), timeout=180) as response:
        for line in response:
            if not line.strip():
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            words.append(piece)
            buffer += piece
            parts = _SENTENCE_END.split(buffer)
            buffer = parts.pop()
            for part in parts:
                if part.strip():
                    say(part.strip())
            if chunk.get("done"):
                break
    if buffer.strip():
        say(buffer.strip())
    return "".join(words).strip()


def warm() -> bool:
    """Load the model into memory now, so the first question is not the slow one."""
    try:
        with urllib.request.urlopen(_request([], False, {}), timeout=180) as response:
            response.read()
    except urllib.error.URLError:
        return False
    return True


def _one_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.removeprefix("->").strip().strip("\"'").strip()


def _talk(messages: list[dict]) -> str:
    lines = []
    for message in messages[1:][-4:]:
        who = "Sir" if message["role"] == "user" else "Friday"
        lines.append(f"{who}: {message['content']}")
    return "\n".join(lines)


def _route(messages: list[dict], text: str) -> str | None:
    """The model's own call: the search words this line needs, or None."""
    talk = _talk(messages[:-1])
    content = (f"Conversation so far:\n{talk}\n\n" if talk else "") + f"Line: {text} ->"
    reply = _one_line(
        ask(
            [{"role": "system", "content": route_prompt()}, {"role": "user", "content": content}],
            temperature=0,
            num_predict=24,
        )
    )
    return None if not reply or reply.upper().startswith("NONE") else reply


def _judge(question: str, results: str) -> str | None:
    """YES when the results answer the question, better search words when they miss, None when stuck."""
    reply = _one_line(
        ask(
            [
                {"role": "system", "content": judge_prompt()},
                {"role": "user", "content": f"Question: {question}\nResults: {results}\nReply:"},
            ],
            temperature=0,
            num_predict=24,
        )
    )
    if reply.upper().startswith("YES"):
        return "YES"
    # The 3B sometimes repeats the instruction before the words, as in "better search words: ...".
    words = reply.rsplit(":", 1)[-1].strip()
    return None if not words or words.upper() in {"NO", "NONE"} else words


def _same_subject(question: str, words: str) -> bool:
    """A new search must share a real word with the question, or the 3B has drifted to its examples."""
    def key(text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}

    return bool(key(question) & key(words))


def _results(query: str) -> str:
    try:
        notes = web_search(query)
    except OSError as error:
        return f"The search failed: {error}"
    return notes or "No results."


def _recent(messages: list[dict]) -> list[dict]:
    return [messages[0], *messages[1:][-RECENT_MESSAGES:]]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _think(messages: list[dict], text: str, offered: str | None, say: Say | None) -> tuple[str, str | None]:
    query = _route(messages, text)
    if query is None:
        return ask(_recent(messages), say), None
    if offered is None and not _ASKED.search(text):
        return f"I don't know that offhand, Sir. Shall I look up {query}?", query

    if say is not None:
        say("One moment, Sir.")
    question = query
    tried: list[str] = []
    found: list[str] = []
    while len(tried) < MAX_SEARCHES:
        if tried and say is not None:
            say("Let me try that another way, Sir.")
        print(f"Looking up: {query}")
        results = _results(query)
        tried.append(query)
        found.append(f"For {query}:\n{results}")
        verdict = _judge(question, results)
        if verdict is None or verdict == "YES" or verdict in tried or not _same_subject(question, verdict):
            break
        query = verdict
    # The results are for this one answer. The conversation keeps only what she said.
    notes = {"role": "system", "content": ANSWER.format(question=question, results="\n\n".join(found))}
    return ask([*_recent(messages), notes], say), None


def handle_turn(
    messages: list[dict],
    text: str,
    offered: str | None,
    say: Say | None = None,
) -> tuple[str | None, str | None]:
    """One thing Sir said. Returns Friday's reply, and the search she offered, if any.

    offered is the search she offered on the last turn. The model reads his
    reply to that offer and decides itself whether it was a yes.
    With say, every word Friday speaks goes through it, a sentence at a time.
    """
    messages[0] = {"role": "system", "content": system_prompt()}
    messages.append({"role": "user", "content": text})
    spoken: list[str] = []

    def out(sentence: str) -> None:
        spoken.append(sentence)
        say(sentence)

    try:
        reply, offer = _think(messages, text, offered, out if say is not None else None)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace").strip()
        print(f"Ollama answered {error.code}. {detail}")
        messages.pop()
        return None, None
    except urllib.error.URLError:
        print("Friday cannot reach Ollama at http://127.0.0.1:11434. Start Ollama, then run this again.")
        messages.pop()
        return None, None
    reply = reply or "Sorry, Sir, I lost my thought. Say that again?"
    messages.append({"role": "assistant", "content": reply})
    if say is not None and not _flat(" ".join(spoken)).endswith(_flat(reply)):
        say(reply)
    return reply, offer


def main() -> None:
    messages = [{"role": "system", "content": system_prompt()}]
    offered: str | None = None
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
        reply, offered = handle_turn(messages, text, offered)
        if reply:
            print(f"Friday: {reply}")


if __name__ == "__main__":
    main()
