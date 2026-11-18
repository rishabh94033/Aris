"""Run Friday.

`python -m friday` types in the terminal.
`python -m friday voice` uses the microphone and the speaker.
"""

import sys

from friday.chat import main as chat_main


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if command == "voice":
        from friday.voice import main as voice_main

        voice_main()
        return
    if command != "chat":
        print("Use: python -m friday")
        print("  or: python -m friday voice")
        raise SystemExit(1)
    chat_main()


if __name__ == "__main__":
    main()
