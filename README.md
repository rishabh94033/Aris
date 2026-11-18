# Friday

Personal assistant of Rishabh Sharma. She addresses Rishabh as Sir.

## Decisions

- Build on this Mac. Later, run the same program on the spare Linux laptop (8 GB RAM, 500 GB SSD).
- The brain is a local model, Ollama `qwen2.5:3b`, so mail and chats stay on our machines.
- Phones and laptops will connect later through Tailscale. They need internet, not the same Wi-Fi.
- CI comes after the spare laptop is running Friday.

## Where we are

Voice on this Mac. If Friday does not know a fact, she asks, and looks it up only after you agree.

Still later: Google Calendar, Teams, Outlook, phones, the spare laptop, CI.

## Run

From this folder:

```bash
.venv/bin/python -m friday voice
```

Say "exit", or press Ctrl+C, to stop. Typing still works with `.venv/bin/python -m friday`.

The ear is local Distil-Whisper `distil-large-v3.5`. It is free and stays on this Mac. The ChatGPT speaking voice needs a key in `.env` in this folder. That file stays off git. Without the key, the voice is Piper's Jenny, a British female voice.

```bash
OPENAI_API_KEY=sk-...
```

The first setup on a machine is:

```bash
brew install python@3.14 portaudio ollama
brew services start ollama
ollama pull qwen2.5:3b
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Voice uses Python 3.14 because those speech libraries do not install on Python 3.15 yet. That setup is already done on this Mac.
