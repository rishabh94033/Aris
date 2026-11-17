"""Terminal conversation with the local model.

Ollama is a program that runs the model and listens on this Mac at
http://127.0.0.1:11434. This file sends the chat there and prints the reply.
"""

import json
import urllib.error
import urllib.request

from aris.persona import system_prompt

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
# The spare Linux laptop has 8 GB of RAM. This 3B model fits there.
MODEL = "qwen2.5:3b"


def ask(messages: list[dict]) -> str:
    """Send the whole conversation so far and return Aris's reply."""
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


def main() -> None:
    messages = [{"role": "system", "content": system_prompt()}]
    print("Aris is here. Type a message, Sir. Type exit to stop.")
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
        messages.append({"role": "user", "content": text})
        # Rewrite the instructions each turn so the clock stays current.
        messages[0] = {"role": "system", "content": system_prompt()}
        try:
            reply = ask(messages)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace").strip()
            print(f"Ollama answered {error.code}.")
            if detail:
                print(detail)
            messages.pop()
            continue
        except urllib.error.URLError:
            print("Aris cannot reach Ollama at http://127.0.0.1:11434.")
            print("Start Ollama, then run this again.")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})
        print(f"Aris: {reply}")


if __name__ == "__main__":
    main()
